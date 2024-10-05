include("affine_dynamics.jl")
include("dataset.jl")
using RobotZoo
using Random
import RobotDynamics as RD

function Phi(model, x,y;vx=0,vy=0)
    input = [x, y, vx .* ones(size(x)), vy .* ones(size(x))]
    input = reduce(hcat,input)'
    return [1 -1] * model(input) 
end

function next_state(A, x, B; vx=0,vy=0, ax=0, ay=0, time=0.1)
    input = [x, y, vx .* ones(size(x)), vy .* ones(size(x))]
    input = reduce(hcat,input)'
    tspan = (0.0, time)
    u = [ax .* ones(size(x)), ay .* ones(size(x))]
    u = reduce(hcat,u)'
    f(x, p, t) = affine_dyn(A, x, B, u)
    prob = ODEProblem(f, input, tspan)
    sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)
    return sol[end]
end

function Phi_elementwise(model, x,y)
    input = [x,y,0,0]
    return ([1 -1] * model(input))[1]
end

function plot_env(X, X_unsafe)
    plt1 = plot(Hyperrectangle(low=low(X)[1:2], high=high(X)[1:2]))
    plot!(plt1, Hyperrectangle(low=low(X_unsafe)[1:2], high=high(X_unsafe)[1:2]), fillcolor=:red)
    return plt1
end

function Phi_naive(model, x,y;vx=0,vy=0)
    input = [x, y, vx .* ones(size(x)), vy .* ones(size(x))]
    input = reduce(hcat,input)'
    return model(input) 
end

function Phi_naive_car(model, x,y;θ=0)
    input = [x, y, θ .* ones(size(x))]
    input = reduce(hcat,input)'
    return model(input) 
end

function Phi_naive_quadrotor(model, x,y;z=2,rot=[1, 0, 0, 0], v=[0,0,0], w=[0,0,0])
    input = [x, y, z .* ones(size(x)),rot[1].* ones(size(x)),rot[2].* ones(size(x)),rot[3].* ones(size(x)),rot[4].* ones(size(x)),
    v[1] .* ones(size(x)),v[2] .* ones(size(x)),v[3] .* ones(size(x)),w[1] .* ones(size(x)),w[2] .* ones(size(x)),w[3] .* ones(size(x))]
    input = reduce(hcat,input)'
    return model(input) 
end

function Phi_naive_quadrotorEuler(model, x,y;z=2,rot=[ 0, 0, 0], v=[0,0,0], w=[0,0,0])
    input = [x, y, z .* ones(size(x)),rot[1].* ones(size(x)),rot[2].* ones(size(x)),rot[3].* ones(size(x)),
    v[1] .* ones(size(x)),v[2] .* ones(size(x)),v[3] .* ones(size(x)),w[1] .* ones(size(x)),w[2] .* ones(size(x)),w[3] .* ones(size(x))]
    input = reduce(hcat,input)'
    return model(input) 
end

function Phi_naive_planar_quadrotor(model, x,y;theta=0, v=[0,0], w=0)
    input = [x, y, theta.* ones(size(x)),v[1].* ones(size(x)),v[2].* ones(size(x)),w.* ones(size(x))]
    input = reduce(hcat,input)'
    return model(input) 
end

function Phi_dot_naive(model, A, B, x,y;vx=0,vy=0, ax=0, ay=0, α=0)
    input = [x,y,vx .* ones(size(x)),vy .* ones(size(x))]
    input = reduce(hcat,input)'
    u = [ax .* ones(size(x)), ay .* ones(size(x))]
    u = reduce(hcat,u)'
    condition = forward_invariance_func(model, A, input, B, u; α=α)
    return condition
end

function Phi_dot_naive_car(model,dyn_model, x,y;θ=0,v=0,w=0,α=0)
    eps = 1e-3
    n,m = RD.dims(dyn_model)
    input = [x,y,θ .* ones(size(x))]
    input = reduce(hcat,input)'
    u = [v .* ones(size(x)), w .* ones(size(x))]
    u = reduce(hcat,u)'
    batchsize = size(input, 2)
    A = []
    B = []
    Δ = []
    for i in 1:batchsize
        z = RD.KnotPoint(input[:, i],u[:, i],0.0,1e-3 ) 
        ∇f = zeros(n, n + m)
        RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), dyn_model, ∇f, zeros(n), z)
        A_ = ∇f[:, 1:n]
        B_ = ∇f[:, n+1:end]
        Δ_ = RobotDynamics.dynamics(dyn_model, input[:, i] .- eps, u[:, i].-eps) - A_ * (input[:, i].-eps) - B_ * (u[:, i] .- eps)
        push!(A, A_)
        push!(B, B_)
        push!(Δ, Δ_)
    end
    A = cat(A..., dims=3)
    B = cat(B..., dims=3)
    Δ = cat(Δ..., dims=2)
    condition = forward_invariance_func(model, A, input, B, u; α=α,Δ=Δ)
    return condition
end

function Phi_dot_naive_planar_quadrotor(model,dyn_model, x,y;theta=0, v=[0,0], w=0,α=0,l=0,u=0)
    eps = 1e-3
    n,m = RD.dims(dyn_model)
    input = [x, y, theta.* ones(size(x)),v[1].* ones(size(x)),v[2].* ones(size(x)),w.* ones(size(x))]
    input = reduce(hcat,input)'
    u = [l .* ones(size(x)), u .* ones(size(x))]
    u = reduce(hcat,u)'
    batchsize = size(input, 2)
    A = []
    B = []
    Δ = []
    for i in 1:batchsize
        z = RD.KnotPoint(input[:, i],u[:, i],0.0,1e-3 ) 
        ∇f = zeros(n, n + m)
        RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), dyn_model, ∇f, zeros(n), z)
        A_ = ∇f[:, 1:n]
        B_ = ∇f[:, n+1:end]
        Δ_ = RobotDynamics.dynamics(dyn_model, input[:, i] .- eps, u[:, i].-eps) - A_ * (input[:, i].-eps) - B_ * (u[:, i] .- eps)
        push!(A, A_)
        push!(B, B_)
        push!(Δ, Δ_)
    end
    A = cat(A..., dims=3)
    B = cat(B..., dims=3)
    Δ = cat(Δ..., dims=2)
    condition = forward_invariance_func(model, A, input, B, u; α=α,Δ=Δ)
    return condition
end

function Phi_dot_naive_quadrotor(model,dyn_model, x,y;z=2,rot=[1, 0, 0, 0], v=[0,0,0], w=[0,0,0],u = [0,0,0,0],α=0)
    eps = 1e-3
    n,m = RD.dims(dyn_model)
    input = [x, y, z .* ones(size(x)),rot[1].* ones(size(x)),rot[2].* ones(size(x)),rot[3].* ones(size(x)),rot[4].* ones(size(x)),
    v[1] .* ones(size(x)),v[2] .* ones(size(x)),v[3] .* ones(size(x)),w[1] .* ones(size(x)),w[2] .* ones(size(x)),w[3] .* ones(size(x))]
    input = reduce(hcat,input)'
    u = [u .* ones(size(x))]
    u = reduce(hcat,u)
    batchsize = size(input, 2)
    A = []
    B = []
    Δ = []
    for i in 1:batchsize
        z = RD.KnotPoint(input[:, i],u[:, i],0.0,1e-3 ) 
        ∇f = zeros(n, n + m)
        RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), dyn_model, ∇f, zeros(n), z)
        A_ = ∇f[:, 1:n]
        B_ = ∇f[:, n+1:end]
        Δ_ = RobotDynamics.dynamics(dyn_model, input[:, i] .- eps, u[:, i].-eps) - A_ * (input[:, i].-eps) - B_ * (u[:, i] .- eps)
        push!(A, A_)
        push!(B, B_)
        push!(Δ, Δ_)
    end
    A = cat(A..., dims=3)
    B = cat(B..., dims=3)
    Δ = cat(Δ..., dims=2)
    condition = forward_invariance_func(model, A, input, B, u; α=α,Δ=Δ)
    return condition
end

function Phi_dot_naive_quadrotorEuler(model,dyn_model, x,y;z=2,rot=[0, 0, 0], v=[0,0,0], w=[0,0,0],u = [0,0,0,0],α=0)
    eps = 1e-3
    input = [x, y, z .* ones(size(x)),rot[1].* ones(size(x)),rot[2].* ones(size(x)),rot[3].* ones(size(x)),
    v[1] .* ones(size(x)),v[2] .* ones(size(x)),v[3] .* ones(size(x)),w[1] .* ones(size(x)),w[2] .* ones(size(x)),w[3] .* ones(size(x))]
    input = reduce(hcat,input)'
    u = [u .* ones(size(x))]
    u = reduce(hcat,u)
    _, input_dot = get_min_u_noAB_vertices(model, input, u, U,zeros(1,1),dyn_model; α=α,same_x=true)
    return forward_invariance_func_noAB(model, input, input_dot; α=α)
end

function Phi_dot_naive_car_vertices(model,dyn_model, x,y, U;θ=0, u = [0,0],α=0)
    eps = 1e-3
    input = [x, y, θ .* ones(size(x))]
    input = reduce(hcat,input)'
    u = [u .* ones(size(x))]
    u = reduce(hcat,u)
    _, input_dot = get_min_u_noAB_vertices(model, input, u, U,zeros(1,1),dyn_model; α=α,same_x=true)
    return forward_invariance_func_noAB(model, input, input_dot; α=α)
end

function h_naive(model,  A, B,U, x,y;vx=0,vy=0, ax=0, ay=0, α=0, lr=0.1,num_iter=100)
    input = [x,y,vx .* ones(size(x)),vy .* ones(size(x))]
    input = reduce(hcat,input)'
    u = [ax .* ones(size(x)), ay .* ones(size(x))]
    u = reduce(hcat,u)'
    _, __, opt_u, opt_condition = verification_forward(model, A, input, B, u, U, α = α, lr=lr,num_iter=num_iter)
    return opt_condition
end

function h_naive_car(model, dyn_model, U, x,y;θ=0,v=0,w=0,α=0, lr=0.1,num_iter=100)
    eps = 1e-3
    n,m = RD.dims(dyn_model)
    input = [x,y,θ .* ones(size(x))]
    input = reduce(hcat,input)'
    u = [v .* ones(size(x)), w .* ones(size(x))]
    u = reduce(hcat,u)'
    batchsize = size(input, 2)
    A = []
    B = []
    Δ = []
    for i in 1:batchsize
        z = RD.KnotPoint(input[:, i],u[:, i],0.0,1e-3 ) 
        ∇f = zeros(n, n + m)
        RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), dyn_model, ∇f, zeros(n), z)
        A_ = ∇f[:, 1:n]
        B_ = ∇f[:, n+1:end]
        Δ_ = RobotDynamics.dynamics(dyn_model, input[:, i] .- eps, u[:, i].-eps) - A_ * (input[:, i].-eps) - B_ * (u[:, i] .- eps)
        push!(A, A_)
        push!(B, B_)
        push!(Δ, Δ_)
    end
    A = cat(A..., dims=3)
    B = cat(B..., dims=3)
    Δ = cat(Δ..., dims=2)
    _, __, opt_u, opt_condition = verification_forward(model, A, input, B, u, U, α = α, lr=lr,num_iter=num_iter,Δ=Δ)
    return opt_condition
end

function h_naive_planar_quadrotor(model, dyn_model, U, x,y;theta=0, v=[0,0], w=0,α=0,l=0,u=0, lr=0.1,num_iter=100)
    eps = 1e-3
    n,m = RD.dims(dyn_model)
    input = [x, y, theta.* ones(size(x)),v[1].* ones(size(x)),v[2].* ones(size(x)),w.* ones(size(x))]
    input = reduce(hcat,input)'
    u = [l .* ones(size(x)), u .* ones(size(x))]
    u = reduce(hcat,u)'
    batchsize = size(input, 2)
    A = []
    B = []
    Δ = []
    for i in 1:batchsize
        z = RD.KnotPoint(input[:, i],u[:, i],0.0,1e-3 ) 
        ∇f = zeros(n, n + m)
        RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), dyn_model, ∇f, zeros(n), z)
        A_ = ∇f[:, 1:n]
        B_ = ∇f[:, n+1:end]
        Δ_ = RobotDynamics.dynamics(dyn_model, input[:, i] .- eps, u[:, i].-eps) - A_ * (input[:, i].-eps) - B_ * (u[:, i] .- eps)
        push!(A, A_)
        push!(B, B_)
        push!(Δ, Δ_)
    end
    A = cat(A..., dims=3)
    B = cat(B..., dims=3)
    Δ = cat(Δ..., dims=2)
    _, __, opt_u, opt_condition = verification_forward(model, A, input, B, u, U, α = α, lr=lr,num_iter=num_iter,Δ=Δ)
    return opt_condition
end

function h_naive_quadrotor(model, dyn_model, U, x,y;z=2,rot=[1, 0, 0, 0], v=[0,0,0], w=[0,0,0],u = [0,0,0,0],α=0, lr=0.1,num_iter=100)
    eps = 1e-3
    n,m = RD.dims(dyn_model)
    input = [x, y, z .* ones(size(x)),rot[1].* ones(size(x)),rot[2].* ones(size(x)),rot[3].* ones(size(x)),rot[4].* ones(size(x)),
    v[1] .* ones(size(x)),v[2] .* ones(size(x)),v[3] .* ones(size(x)),w[1] .* ones(size(x)),w[2] .* ones(size(x)),w[3] .* ones(size(x))]
    input = reduce(hcat,input)'
    u = [u .* ones(size(x))]
    u = reduce(hcat,u)
    batchsize = size(input, 2)
    A = []
    B = []
    Δ = []
    for i in 1:batchsize
        z = RD.KnotPoint(input[:, i],u[:, i],0.0,1e-3 ) 
        ∇f = zeros(n, n + m)
        RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), dyn_model, ∇f, zeros(n), z)
        A_ = ∇f[:, 1:n]
        B_ = ∇f[:, n+1:end]
        Δ_ = RobotDynamics.dynamics(dyn_model, input[:, i] .- eps, u[:, i].-eps) - A_ * (input[:, i].-eps) - B_ * (u[:, i] .- eps)
        push!(A, A_)
        push!(B, B_)
        push!(Δ, Δ_)
    end
    A = cat(A..., dims=3)
    B = cat(B..., dims=3)
    Δ = cat(Δ..., dims=2)
    _, __, opt_u, opt_condition = verification_forward(model, A, input, B, u, U, α = α, lr=lr,num_iter=num_iter,Δ=Δ)
    return opt_condition
end

function Phi_elementwise_naive(model, x,y)
    input = [x,y,0,0]
    return (model(input))[1]
end
