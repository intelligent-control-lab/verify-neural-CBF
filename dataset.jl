using DifferentialEquations
using LazySets

using ProgressBars
using JLD2


function find_ada_input_area(U::Hyperrectangle, random_point)
    if length(random_point) != 4 || dim(U) != 2
        println("only support 2D double-integrator")
        return U
    end
    random_point[3] > 0 ? ux_min = low(U, 1) : ux_min = 0
    random_point[3] > 0 ? ux_max = 0 : ux_max = high(U, 1)
    random_point[4] > 0 ? uy_min = low(U, 2) : uy_min = 0
    random_point[4] > 0 ? uy_max = 0 : uy_max = high(U, 2)
    output = Hyperrectangle(low=[ux_min, uy_min], high=[ux_max, uy_max])
    return output
    
end

function find_ada_non_admissible_area(non_admissible_area::Hyperrectangle, random_point, input_bound)
    if length(random_point) != 4 || dim(non_admissible_area) != 4 || dim(input_bound) != 2
        println("only support 2D double-integrator")
        return non_admissible_area
    end
    x_min = low(non_admissible_area, 1) + max(0, random_point[3]) * random_point[3] / (2 * low(input_bound, 1))
    x_max = high(non_admissible_area, 1) + max(0, -random_point[3]) * (-random_point[3]) / (2 * high(input_bound, 1))
    y_min = low(non_admissible_area, 2) + max(0, random_point[4]) * random_point[4] / (2 * low(input_bound, 2))
    y_max = high(non_admissible_area, 2) + max(0, -random_point[4]) * (-random_point[4]) / (2 * high(input_bound, 2))
    # @show non_admissible_area
    @assert x_min ≤ low(non_admissible_area, 1)
    @assert y_min ≤ low(non_admissible_area, 2)
    @assert x_max ≥ high(non_admissible_area, 1)
    @assert y_max ≥ high(non_admissible_area, 2)
    output = Hyperrectangle(low=[x_min, y_min, low(non_admissible_area, 3),low(non_admissible_area, 3)], high=[x_max, y_max, high(non_admissible_area,3), high(non_admissible_area,4)])
    # @show random_point, output
    return output
end


function random_point_in_hyperrectangle(hyperrectangle::Hyperrectangle, non_admissible_area=nothing, input_bound=nothing)
    dimensions = dim(hyperrectangle)
    # while true
    random_point = Vector{Float32}(undef, dimensions)
    for i in 1:dimensions
        random_point[i] = rand() * (high(hyperrectangle, i)-low(hyperrectangle, i)) + low(hyperrectangle, i)
    end
    isnothing(non_admissible_area) && return random_point, true
    # @show non_admissible_area, random_point, input_bound
    ada_non_admissible_area = find_ada_non_admissible_area(non_admissible_area, random_point, input_bound)
    (random_point ∉ ada_non_admissible_area) && return random_point, true
    return random_point, false
        # continue
    # end
end

function affine_dyn(A::AbstractMatrix, x::AbstractArray, B::AbstractMatrix, u::AbstractArray)
    ẋ = A * x + B * u
    return ẋ
end

