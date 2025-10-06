import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
from collect_data import DubinsCar, Hyperrectangle

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# class DubinsCar:
#     def __init__(self):
#         self.state_dim = 3  # state dimension
#         self.control_dim = 2  # control dimension
        
#     def dynamics(self, x, u):
#         """
#         Dynamics for Dubins Car: [ẋ, ẏ, θ̇] = [v*cos(θ), v*sin(θ), ω]
#         where v is forward velocity and ω is angular velocity
#         """
#         v = u[0]
#         w = u[1]
#         theta = x[2]
        
#         dx = torch.zeros_like(x)
#         dx[0] = v * torch.cos(theta)  # ẋ = v*cos(θ)
#         dx[1] = v * torch.sin(theta)  # ẏ = v*sin(θ)
#         dx[2] = w                     # θ̇ = ω
        
#         return dx
    
#     def jacobian(self, x, u):
#         """Compute the Jacobian of dynamics with respect to state and control"""
#         batch_size = x.shape[0]
#         A = torch.zeros(batch_size, self.n, self.n, device=x.device)
#         B = torch.zeros(batch_size, self.n, self.m, device=x.device)
        
#         # State Jacobian (A)
#         v = u[:, 0]
#         theta = x[:, 2]
        
#         # ∂ẋ/∂θ = -v*sin(θ)
#         A[:, 0, 2] = -v * torch.sin(theta)
        
#         # ∂ẏ/∂θ = v*cos(θ)
#         A[:, 1, 2] = v * torch.cos(theta)
        
#         # Control Jacobian (B)
#         # ∂ẋ/∂v = cos(θ)
#         B[:, 0, 0] = torch.cos(theta)
        
#         # ∂ẏ/∂v = sin(θ)
#         B[:, 1, 0] = torch.sin(theta)
        
#         # ∂θ̇/∂ω = 1
#         B[:, 2, 1] = 1.0
        
#         return A, B

# class Hyperrectangle:
#     def __init__(self, low, high):
#         self.low = torch.tensor(low, dtype=torch.float32)
#         self.high = torch.tensor(high, dtype=torch.float32)
    
#     def vertices_list(self):
#         """Generate all vertices of the hyperrectangle"""
#         n = len(self.low)
#         vertices = []
        
#         # Generate all combinations of low and high values
#         for i in range(2**n):
#             vertex = torch.zeros_like(self.low)
#             for j in range(n):
#                 if (i >> j) & 1:
#                     vertex[j] = self.high[j]
#                 else:
#                     vertex[j] = self.low[j]
#             vertices.append(vertex)
            
#         return vertices

def f_batch(A, x):
    """
    Equivalent to batched_mul(A, x) in Julia
    A: [batch_size, state_dim, state_dim] or [state_dim, state_dim]
    x: [batch_size, state_dim]
    """
    if len(A.shape) == 2:
        # If A is a matrix, broadcast it to all batches
        return torch.matmul(A, x.unsqueeze(-1)).squeeze(-1)
    else:
        # If A is already batched, use batch matrix multiplication
        return torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)

def g_batch(B, u):
    """
    Equivalent to batched_mul(B, u) in Julia
    B: [batch_size, state_dim, control_dim] or [state_dim, control_dim]
    u: [batch_size, control_dim]
    """
    if len(B.shape) == 2:
        # If B is a matrix, broadcast it to all batches
        return torch.matmul(B, u.unsqueeze(-1)).squeeze(-1)
    else:
        # If B is already batched, use batch matrix multiplication
        return torch.bmm(B, u.unsqueeze(-1)).squeeze(-1)

def affine_dyn_batch(A, x, B, u, Delta=None):
    """
    Compute the affine dynamics: ẋ = Ax + Bu + Δ
    """
    f_x = f_batch(A, x)
    g_u = g_batch(B, u)
    x_dot = f_x + g_u
    
    if Delta is not None:
        x_dot = x_dot + Delta
        
    return x_dot

def forward_invariance_func(phi, A, x, B, u, alpha=0, Delta=None):
    """
    Compute the time derivative of phi along system trajectories plus a decay term: ϕ̇ + α*ϕ
    """
    batch_size = x.shape[0]
    state_dim = x.shape[1]
    
    # We need a clone of x with requires_grad=True for computing gradients
    if torch.is_tensor(x):
        x_grad = x.clone().detach().requires_grad_(True)
    else:
        x = torch.tensor(x,dtype=torch.float32)
        x_grad = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    if not torch.is_tensor(u):
        u = torch.tensor(u,dtype=torch.float32)
    if not torch.is_tensor(A):
        A = torch.tensor(A,dtype=torch.float32)
    if not torch.is_tensor(B):
        B = torch.tensor(B,dtype=torch.float32)
    if Delta is not None and not torch.is_tensor(Delta):
        Delta = torch.tensor(Delta,dtype=torch.float32)
    # Compute phi(x)
    phi_x = phi(x_grad)
    
    # Compute gradient of phi with respect to x
    ones = torch.ones_like(phi_x, device=x.device)
    gradients = torch.autograd.grad(
        outputs=phi_x, 
        inputs=x_grad,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute state derivatives
    x_dot = affine_dyn_batch(A, x, B, u, Delta=Delta)
    
    # Compute ϕ̇ = ∇ϕᵀẋ (dot product of gradient and dynamics)
    phi_dot = torch.sum(gradients * x_dot, dim=1, keepdim=True)
    
    # Compute ϕ̇ + α*ϕ
    l = phi_dot + alpha * phi(x)
    
    return l

def forward_invariance_func_noAB(phi, x, x_dot, alpha=0):
    """
    Compute the time derivative of phi along given trajectories plus a decay term: ϕ̇ + α*ϕ
    """
    batch_size = x.shape[0]
    state_dim = x.shape[1]
    
    # We need a clone of x with requires_grad=True for computing gradients
    x_grad = x.clone().detach().requires_grad_(True)
    
    # Compute phi(x)
    phi_x = phi(x_grad)
    
    # Compute gradient of phi with respect to x
    ones = torch.ones_like(phi_x, device=x.device)
    gradients = torch.autograd.grad(
        outputs=phi_x, 
        inputs=x_grad,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute ϕ̇ = ∇ϕᵀẋ (dot product of gradient and dynamics)
    phi_dot = torch.sum(gradients * x_dot, dim=1, keepdim=True)
    
    # Compute ϕ̇ + α*ϕ
    l = phi_dot + alpha * phi(x)
    
    return l

def relu(x):
    """ReLU activation function"""
    return torch.maximum(x, torch.zeros_like(x))

def sigmoid_fast(x):
    """Fast approximation of sigmoid function"""
    return torch.sigmoid(x)

def loss_naive_safeset(phi, x, y_init):
    """
    Compute the naive safe set loss
    """
    # Extract first dimension of y_init (safe: 1; unsafe: 0)
    y_init = y_init.squeeze(1)  # [batch_size]
    
    # Compute phi(x)
    phi_x = phi(x).squeeze(1)  # [batch_size]
    
    # (2*y_init - 1) maps safe to 1 and unsafe to -1
    # We want phi(x) to be negative for safe states and positive for unsafe states
    # loss is positive when constraint is violated
    loss = relu((2 * y_init - 1) * phi_x + 1e-6)
    
    return torch.mean(loss)

def loss_regularization(phi, x, y_init):
    """
    Compute regularization loss to encourage smooth boundaries
    """
    # Extract first dimension of y_init (safe: 1; unsafe: 0)
    y_init = y_init.squeeze(1)  # [batch_size]
    
    # Compute phi(x)
    phi_x = phi(x).squeeze(1)  # [batch_size]
    
    # Sigmoid of the constraint
    loss = sigmoid_fast((2 * y_init - 1) * phi_x)
    
    return torch.mean(loss)

def loss_naive_fi(phi, A, x, B, u, y_init, use_pgd=False, use_adv=False, alpha=0, lr=1, num_iter=10, epsilon=0.1, Delta=None):
    """
    Compute forward invariance loss for the CBF
    """
    # Extract first dimension of y_init (safe: 1; unsafe: 0)
    y_init = y_init.squeeze(1)  # [batch_size]
    
    # Find indices of safe states
    safe_indices = (y_init == 1).nonzero(as_tuple=True)[0]
    
    if len(safe_indices) == 0:
        return torch.tensor(0.0, device=x.device)
    
    # Select only safe states
    x_safe = x[safe_indices]
    u_safe = u[safe_indices]
    
    if Delta is not None:
        if len(A.shape) == 3:  # Batched A
            A_safe = A[safe_indices]
        else:
            A_safe = A  # Use the same A for all samples if not batched
            
        if len(B.shape) == 3:  # Batched B
            B_safe = B[safe_indices]
        else:
            B_safe = B  # Use the same B for all samples if not batched
            
        Delta_safe = Delta[safe_indices]
    else:
        A_safe = A
        B_safe = B
        Delta_safe = None
    
    # Find states close to the boundary (|phi(x)| < epsilon)
    with torch.no_grad():
        phi_x = phi(x_safe)
    
    boundary_indices = torch.abs(phi_x) < epsilon
    boundary_indices = boundary_indices.squeeze(1).nonzero(as_tuple=True)[0]
    
    if len(boundary_indices) == 0:
        return torch.tensor(0.0, device=x.device)
    
    # Select only boundary states
    x_boundary = x_safe[boundary_indices]
    u_boundary = u_safe[boundary_indices]
    
    if Delta is not None:
        if len(A.shape) == 3:  # Batched A
            A_boundary = A_safe[boundary_indices]
        else:
            A_boundary = A_safe
            
        if len(B.shape) == 3:  # Batched B
            B_boundary = B_safe[boundary_indices]
        else:
            B_boundary = B_safe
            
        Delta_boundary = Delta_safe[boundary_indices]
    else:
        A_boundary = A_safe
        B_boundary = B_safe
        Delta_boundary = None
    
    # If requested, use adversarial examples
    if use_adv:
        # Implement pgd_find_x_notce equivalent here if needed
        pass
    
    # If requested, find best-case control inputs
    u_best = u_boundary
    if use_pgd:
        u_best = pgd_find_u_notce(phi, A_boundary, x_boundary, B_boundary, u_boundary, U, 
                                 alpha=alpha, lr=lr, num_iter=num_iter, Delta=Delta_boundary)
    
    # Compute forward invariance constraint
    fi_values = forward_invariance_func(phi, A_boundary, x_boundary, B_boundary, u_best, 
                                     alpha=alpha, Delta=Delta_boundary)
    
    # Loss is positive when constraint is violated (fi_values > 0)
    loss = relu(fi_values + 1e-6)
    
    return torch.mean(loss)

def pgd_find_u_notce(phi, A, x, B, u_0, U, alpha=0, lr=1, num_iter=10, Delta=None):
    """
    Use projected gradient descent to find the best-case control input that minimizes the CBF derivative
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x,dtype=torch.float32)
    if not torch.is_tensor(u_0):
        u_0 = torch.tensor(u_0,dtype=torch.float32)
    if not torch.is_tensor(A):
        A = torch.tensor(A,dtype=torch.float32)
    if not torch.is_tensor(B):
        B = torch.tensor(B,dtype=torch.float32)
    # Make a detached copy with requires_grad=True
    u = u_0.clone().detach().requires_grad_(True)
    
    # Ensure we have the bounds on the right device
    if torch.is_tensor(U.low):
        low_U = U.low.to(u.device)
        high_U = U.high.to(u.device)
    else:
        low_U = torch.tensor(U.low,dtype=torch.float32).to(u.device)
        high_U = torch.tensor(U.high,dtype=torch.float32).to(u.device)
    
    # Note: in the original Julia code, this is done manually, but we use an optimizer
    optimizer = optim.SGD([u], lr=lr)
    
    for i in range(num_iter):
        # Zero gradients
        optimizer.zero_grad()
        
        # Compute forward invariance constraint
        # Note: We make sure x doesn't require gradients inside forward_invariance_func
        fi_values = forward_invariance_func(phi, A, x, B, u, alpha=alpha, Delta=Delta)
        
        # We want to minimize the constraint value (find u that makes it most positive)
        loss = torch.mean(fi_values)
        
        # Compute gradients
        loss.backward(retain_graph=True)
        
        # Update u
        optimizer.step()
        
        # Project u back to the constraint set
        with torch.no_grad():
            u.data = torch.clamp(u.data, min=low_U.expand_as(u.data), max=high_U.expand_as(u.data))
            
    # Return the detached tensor
    return u.detach()

def pgd_find_x_notce(phi, A, x_0, B, u, X_list, alpha=0, lr=0.01, num_iter=10, Delta=None):
    """
    Use projected gradient descent to find the worst-case state that maximizes the CBF derivative
    """
    # Make a detached copy with requires_grad=True
    x = x_0.clone().detach().requires_grad_(True)
    
    # Extract low and high bounds for each state
    low_X = torch.stack([X.low for X in X_list]).to(x.device)
    high_X = torch.stack([X.high for X in X_list]).to(x.device)
    
    optimizer = optim.SGD([x], lr=lr)
    
    for i in range(num_iter):
        # Zero gradients
        optimizer.zero_grad()
        
        # Compute forward invariance constraint
        fi_values = forward_invariance_func(phi, A, x, B, u, alpha=alpha, Delta=Delta)
        
        # We want to maximize the constraint value (find x that makes it most positive)
        loss = -torch.mean(fi_values)
        
        # Compute gradients
        loss.backward(retain_graph=True)
        
        # Update x
        optimizer.step()
        
        # Project x back to the constraint set
        with torch.no_grad():
            for j in range(x.shape[0]):
                x.data[j] = torch.clamp(x.data[j], min=low_X[j], max=high_X[j])
    
    return x.detach()

# Define the model
class CBFModel(nn.Module):
    def __init__(self, state_dim):
        super(CBFModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Define hyperrectangles
X = Hyperrectangle(low=[0, 0, 0], high=[4, 4, np.pi],npy=False)
U = Hyperrectangle(low=[-1, -1], high=[1, 1],npy=False)
X_unsafe = Hyperrectangle(low=[1.5, 0, 0], high=[2.5, 2, np.pi],npy=False)

# Function to load data
def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# Load training and test data
raw_training_data = load_data("car_training_data.pkl")
raw_test_data = load_data("car_test_data.pkl")

# Prepare data for PyTorch
def prepare_data(raw_data):
    x_data = torch.tensor(np.column_stack([d[0] for d in raw_data]), dtype=torch.float32).to(device)
    u_data = torch.tensor(np.column_stack([d[1] for d in raw_data]), dtype=torch.float32).to(device)
    y_data = torch.tensor(np.array([d[2] for d in raw_data]), dtype=torch.float32).to(device)
    return TensorDataset(x_data.t()[:1000,:], u_data.t()[:1000,:], y_data.reshape(1, -1).t()[:1000,:])


# Main training function
def train_cbf():
    # Hyperparameters
    batchsize = 128
    lambda_param = 1.0
    mu = 0.1  # regularization weight
    alpha = 0.0
    use_pgd = True
    
    # Learning rate parameters
    ini_lr = 0.01
    lr_decay_rate = 0.2
    lr_decay_epoch = 4
    total_epoch = 20
    
    # Initialize model
    state_dim = 3  # [x, y, theta] for DubinsCar
    model = CBFModel(state_dim).to(device)
    
    # Initialize optimizer
    optimizer = optim.NAdam(
        model.parameters(),
        lr=ini_lr,
        betas=(0.9, 0.999),
        weight_decay=0.0
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_decay_epoch,
        gamma=lr_decay_rate
    )
    
    # Create car dynamics model
    dyn_model = DubinsCar()
    
    # Prepare data
    training_dataset = prepare_data(raw_training_data)
    test_dataset = prepare_data(raw_test_data)
    
    train_loader = DataLoader(training_dataset, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)
    
    # Lists to store loss history
    training_losses = []
    test_losses = []
    
    # Small epsilon for numerical stability
    eps = 1e-3
    
    for epoch in range(1, total_epoch + 1):
        model.train()
        training_loss_epoch = []
        
        # Training loop
        for x_batch, u_batch, y_init_batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            x_batch = x_batch.to(device)
            u_batch = u_batch.to(device)
            y_init_batch = y_init_batch.to(device)
            
            # Compute linearized dynamics (A, B matrices) for each point in batch
            batch_size = x_batch.shape[0]
            A_batch = torch.zeros(batch_size, dyn_model.state_dim, dyn_model.state_dim, device=device)
            B_batch = torch.zeros(batch_size, dyn_model.state_dim, dyn_model.control_dim, device=device)
            Delta = torch.zeros(batch_size, dyn_model.state_dim, device=device)
            
            # Get linearized dynamics for each state-control pair
            A_matrices, B_matrices = dyn_model.jacobian(x_batch, u_batch)
            A_batch = A_matrices
            B_batch = B_matrices
            
            # Compute residual term (Δ)
            for i in range(batch_size):
                x_eps = x_batch[i] - eps
                u_eps = u_batch[i] - eps
                
                # Compute actual dynamics
                f_actual = dyn_model.dynamics(x_eps, u_eps,npy=False)
                
                # Compute linearized dynamics
                f_linear = torch.matmul(A_batch[i], x_eps) + torch.matmul(B_batch[i], u_eps)
                
                # Residual is the difference
                Delta[i] = f_actual - f_linear
            
            # Create a copy of u_batch that we can modify
            u_best = u_batch.clone()#.detach()
            
            # If using PGD, find best-case control inputs
            if use_pgd:
                # with torch.no_grad():
                u_best = pgd_find_u_notce(model, A_batch, x_batch, B_batch, u_batch, U, alpha=alpha, Delta=Delta)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute losses
            safe_loss = loss_naive_safeset(model, x_batch, y_init_batch)
            fi_loss = loss_naive_fi(model, A_batch, x_batch, B_batch, u_best, y_init_batch, use_pgd=False, alpha=alpha, Delta=Delta)
            reg_loss = loss_regularization(model, x_batch, y_init_batch)
            
            # Total loss
            loss = safe_loss + lambda_param * fi_loss + mu * reg_loss
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Record loss
            training_loss_epoch.append(loss.item())
        
        # Update learning rate
        scheduler.step()
        
        # Evaluate on test set
        model.eval()
        test_loss_epoch = []
        
        # with torch.no_grad():
        for x_batch, u_batch, y_init_batch in tqdm(test_loader, desc=f"Testing Epoch {epoch}"):
            x_batch = x_batch.to(device)
            u_batch = u_batch.to(device)
            y_init_batch = y_init_batch.to(device)
            
            # Compute linearized dynamics
            A_batch, B_batch = dyn_model.jacobian(x_batch, u_batch)
            
            # Compute residual term (Δ)
            batch_size = x_batch.shape[0]
            Delta = torch.zeros(batch_size, dyn_model.state_dim, device=device)
            
            for i in range(batch_size):
                x_eps = x_batch[i] - eps
                u_eps = u_batch[i] - eps
                
                # Compute actual dynamics
                f_actual = dyn_model.dynamics(x_eps, u_eps,npy=False)
                
                # Compute linearized dynamics
                f_linear = torch.matmul(A_batch[i], x_eps) + torch.matmul(B_batch[i], u_eps)
                
                # Residual is the difference
                Delta[i] = f_actual - f_linear
            
            # Compute losses
            safe_loss = loss_naive_safeset(model, x_batch, y_init_batch)
            fi_loss = loss_naive_fi(model, A_batch, x_batch, B_batch, u_batch, y_init_batch, use_pgd=use_pgd, alpha=alpha, Delta=Delta)
            reg_loss = loss_regularization(model, x_batch, y_init_batch)
            
            # Total loss
            test_loss = safe_loss + lambda_param * fi_loss + mu * reg_loss
            test_loss_epoch.append(test_loss.item())
        
        # Save model
        torch.save(model.state_dict(), f"car_naive_model_1_0_0.1_pgd_relu_{epoch}.pt")
        
        # Compute average losses
        avg_train_loss = sum(training_loss_epoch) / len(training_loss_epoch)
        avg_test_loss = sum(test_loss_epoch) / len(test_loss_epoch)
        
        # Record average losses
        training_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        print(f"Epoch {epoch}, Training Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
    
    # Plot training and test losses
    plot_loss(training_losses, test_losses)
    
    return model, training_losses, test_losses

def plot_loss(train_loss, test_loss, xlabel="Epoch", ylabel="Loss", title="Training and Test Loss"):
    """Plot training and test loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Training")
    plt.plot(test_loss, label="Test")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curves.png")
    plt.show()

# Run the training
if __name__ == "__main__":
    model, training_losses, test_losses = train_cbf()
