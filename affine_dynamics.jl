using Flux
using LinearAlgebra
using Zygote
using ReverseDiff


function f_batch(A::Union{AbstractMatrix,AbstractArray}, x::AbstractArray)
    isa(A, AbstractMatrix) && return batched_mul(A, x)
    @assert size(x, 2) == size(A, 3) 
    @assert length(size(A)) == 3
    x = reshape(x, (size(x, 1), 1,size(x, 2)))
    return dropdims(A ⊠ x, dims=2)
    
end

function g_batch(B::Union{AbstractMatrix,AbstractArray}, u::AbstractArray)
    isa(B, AbstractMatrix) && return batched_mul(B, u)
    @assert size(u, 2) == size(B, 3) 
    @assert length(size(B)) == 3
    u = reshape(u, (size(u, 1), 1,size(u, 2)))
    return dropdims(B ⊠ u, dims=2)
end

function affine_dyn_batch(A::Union{AbstractMatrix,AbstractArray}, x::AbstractArray, B::Union{AbstractMatrix,AbstractArray}, u::AbstractArray;Δ=nothing)
    f_x = f_batch(A, x)
    g_u = g_batch(B, u)
    ẋ = f_x + g_u
    isnothing(Δ) && (Δ = zeros(size(ẋ)))
    return ẋ + Δ 
end

function forward_invariance_func(ϕ::Chain, A::Union{AbstractMatrix,AbstractArray}, x::AbstractArray, B::Union{AbstractMatrix,AbstractArray}, u::AbstractArray; α=0,Δ=nothing)
    state_dim, batchsize = size(x)
    ẋ = affine_dyn_batch(A, x, B, u;Δ=Δ)
    ẋ = reshape(ẋ, (state_dim, 1, batchsize))
    _, ∇ϕ = Zygote.pullback(ϕ, x)
    ∇ϕ_x = ∇ϕ(ones(size(x)))[1] ./ state_dim
    ∇ϕ_x = reshape(∇ϕ_x, (1, state_dim, batchsize))
    ϕ̇ = reshape(batched_mul(∇ϕ_x, ẋ), size(ϕ(x)))
    l = ϕ̇ .+ α .* ϕ(x)
    return l
end

function forward_invariance_func_noAB(ϕ::Chain,  x::AbstractArray, ẋ::AbstractArray; α=0)
    state_dim, batchsize = size(x)
    ẋ = reshape(ẋ, (state_dim, 1, batchsize))
    _, ∇ϕ = Zygote.pullback(ϕ, x)
    ∇ϕ_x = ∇ϕ(ones(size(x)))[1] ./ state_dim
    ∇ϕ_x = reshape(∇ϕ_x, (1, state_dim, batchsize))
    ϕ̇ = reshape(batched_mul(∇ϕ_x, ẋ), size(ϕ(x)))
    l = ϕ̇ .+ α .* ϕ(x)
    return l
end


function loss_safe_set(ϕ::Chain, x::AbstractArray,y_init::AbstractArray)
    return Flux.Losses.mse(max.(0, (2 .* y_init .- 1) .* ϕ(x)), 0)
end


function loss_naive_safeset(ϕ::Chain, x::AbstractArray,y_init::AbstractArray)
    y_init = y_init[1, :] # safe: 1; unsafe: 0
    loss = relu((2 .* y_init .- 1) .* ϕ(x)[1, :] .+ 1e-6)
    return sum(loss) / size(loss)[end]
end

function loss_regularization(ϕ::Chain, x::AbstractArray,y_init::AbstractArray)
    y_init = y_init[1, :] # safe: 1; unsafe: 0
    loss = sigmoid_fast((2 .* y_init .- 1) .* ϕ(x)[1, :])
    return sum(loss) / size(loss)[end]
end

function loss_forward_invariance(ϕ::Chain, A::Union{AbstractMatrix,AbstractArray}, x::AbstractArray, B::Union{AbstractMatrix,AbstractArray}, u::AbstractArray, y_cbf::AbstractArray; α=0,Δ=nothing)
    l = forward_invariance_func(ϕ, A, x, B, u; α,Δ=Δ)
    return Flux.Losses.mse(max.(0, l), 0)
end

function loss_naive_fi(ϕ::Chain, A::Union{AbstractMatrix,AbstractArray}, x::AbstractArray, B::Union{AbstractMatrix,AbstractArray}, u::AbstractArray, y_init::AbstractArray; use_pgd=false, use_adv = false, α=0, lr =1, num_iter=10,ϵ=0.1,Δ=nothing)
    y_init = y_init[1, :]
    index = findall(x->x==1, y_init)
    
    size(index)[1] == 0 && return 0

    x = x[:, index]
    u = u[:, index]

    if !isnothing(Δ)
        A = A[:,:, index]
        B = B[:,:, index]
        Δ = Δ[:, index]
    end

    @assert α==0
    
    mask = abs.(ϕ(x)) .< ϵ
    index = findall(x->x==true, mask[1,:])
    size(index)[1] == 0 && return 0
    x = x[:, index]
    u = u[:, index]
    if !isnothing(Δ)
        A = A[:,:, index]
        B = B[:,:, index]
        Δ = Δ[:, index]
    end
    if use_adv
        X_lcoal = [Hyperrectangle(x[:, i], radius_hyperrectangle(X) ./ 20) for i=1:size(x)[2]]
        x = pgd_find_x_notce(ϕ, A, x, B, u, X_lcoal; α = α,Δ=Δ)
    end
    use_pgd && (u = pgd_find_u_notce(ϕ, A, x, B, u, U; α = α, lr =lr, num_iter=num_iter,Δ=Δ))
    loss = relu(forward_invariance_func(ϕ, A, x, B, u; α,Δ=Δ) .+ 1e-6)
    return sum(loss) / size(loss)[end]
end

function get_min_u_noAB_vertices(ϕ::Chain, x::AbstractArray, u::AbstractArray, U::Hyperrectangle,y_init::AbstractArray, dyn_model; use_pgd=false, α=0,use_adv = false, ϵ=0.1, same_x=false)
    if !same_x
        y_init = y_init[1, :]
        index = findall(x->x==1, y_init)
        
        size(index)[1] == 0 && return nothing, nothing

        x = x[:, index]
        u = u[:, index]


        @assert α==0
        
        mask = abs.(ϕ(x)) .< ϵ
        index = findall(x->x==true, mask[1,:])
        
        size(index)[1] == 0 && return nothing, nothing
        x = x[:, index]
        u = u[:, index]
    end
    u_cand = vertices_list(U)
    
    ẋ_batch = zeros(size(x))
    for i in 1:size(x, 2)
        min_ϕ̇ = Inf
        min_ẋ = nothing
        for j in 1:length(u_cand)
            
            cand_ẋ = dyn_model(x[:, i], u_cand[j])
            
            if min_ϕ̇ > forward_invariance_func_noAB(ϕ,x[:, i:i],cand_ẋ; α)[1, 1]
                min_ϕ̇ = forward_invariance_func_noAB(ϕ,x[:, i:i],cand_ẋ; α)[1, 1]
                min_ẋ = cand_ẋ
            end
        end
        ẋ_batch[:, i] .= min_ẋ
    end
    return x,ẋ_batch
end

function loss_naive_fi_noAB(ϕ::Chain, x::AbstractMatrix, ẋ_batch::AbstractMatrix; α=0)
    loss = relu(forward_invariance_func_noAB(ϕ,x,ẋ_batch; α) .+ 1e-6)
    return sum(loss) / size(loss)[end]
end




function verification_forward(ϕ::Chain, A::Union{AbstractMatrix,AbstractArray}, x::AbstractArray, B::Union{AbstractMatrix,AbstractArray}, u_0::AbstractArray, U::Hyperrectangle; α=0, lr = 1,num_iter=10,Δ=nothing)

    original_condition = (forward_invariance_func(ϕ, A, x, B, u_0; α,Δ=Δ) .≤ 0)
    u = pgd_find_u_notce(ϕ, A, x, B, u_0, U; α = α, lr =lr, num_iter=num_iter,Δ=Δ)
    return original_condition, forward_invariance_func(ϕ, A, x, B, u; α,Δ=Δ) .≤ 0, u, forward_invariance_func(ϕ, A, x, B, u; α,Δ=Δ)
end


function pgd_find_u_notce(ϕ::Chain, A::Union{AbstractMatrix,AbstractArray}, x::AbstractArray, B::Union{AbstractMatrix,AbstractArray}, u_0::AbstractArray, U::Hyperrectangle; α=0, lr = 1,num_iter=10,Δ=nothing)
    u = u_0
    for i in 1:num_iter
        function l_min_u_function(u1::AbstractArray)
            return forward_invariance_func(ϕ, A, x, B, u1; α,Δ=Δ)
        end
        

        val, ∇l = Zygote.pullback(l_min_u_function, u)
        state_dim, batchsize = size(u)
        ∇l_u = ∇l(ones(size(u)))[1] ./ state_dim
        ∇l_u = reshape(∇l_u, (state_dim, batchsize))
        u_old = copy(u)
        u = u - lr .* ∇l_u
        u = low(U) .+ relu(u .- low(U))
        u = high(U) .- relu(high(U) .- u)
        if u == u_old
            break
        end
    end
    return u
end

function pgd_find_x_notce(ϕ::Chain, A::Union{AbstractMatrix,AbstractArray}, x_0::AbstractArray, B::Union{AbstractMatrix,AbstractArray}, u::AbstractArray, X::Vector; α=0, lr = 0.01,num_iter=10,Δ=nothing)
    x = x_0
    low_X = [low(X[i]) for i = 1:length(X)]
    high_X = [high(X[i]) for i = 1:length(X)]
    low_X = reduce(hcat, low_X)
    high_X = reduce(hcat, high_X)
    for i in 1:num_iter
        function l_max_x_function(x1::AbstractArray)
            return forward_invariance_func(ϕ, A, x1, B, u; α,Δ=Δ)
        end
        

        val, ∇l = Zygote.pullback(l_max_x_function, x)
        state_dim, batchsize = size(x)
        ∇l_x = ∇l(ones(size(x)))[1] ./ state_dim
        ∇l_x = reshape(∇l_x, (state_dim, batchsize))
        x_old = copy(x)
        x = x + lr .* ∇l_x
        x = low_X .+ relu(x .- low_X)
        x = high_X .- relu(high_X .- x)
        if x == x_old
            break
        end
    end
    return x
end


function pgd_find_x_notce_noAB(ϕ::Chain, x_0::AbstractArray, X::Vector,ẋ::AbstractArray; α=0, lr = 0.01,num_iter=10)
    x = x_0
    low_X = [low(X[i]) for i = 1:length(X)]
    high_X = [high(X[i]) for i = 1:length(X)]
    low_X = reduce(hcat, low_X)
    high_X = reduce(hcat, high_X)
    for i in 1:num_iter
        function l_max_x_function(x1::AbstractArray)
            return forward_invariance_func_noAB(ϕ, x1,ẋ; α)
        end
        

        val, ∇l = Zygote.pullback(l_max_x_function, x)
        state_dim, batchsize = size(x)

        ∇l_x = ∇l(ones(size(x)))[1] ./ state_dim
        ∇l_x = reshape(∇l_x, (state_dim, batchsize))
        x_old = copy(x)
        x = x + lr .* ∇l_x
        x = low_X .+ relu(x .- low_X)
        x = high_X .- relu(high_X .- x)
        if x == x_old
            break
        end
    end
    return x
end