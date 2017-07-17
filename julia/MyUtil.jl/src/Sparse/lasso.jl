export LassoSolver
@compat abstract type LassoSolver end

export fit!

"""
fit!(solver, [x::Vector,] y::Vector, A::AbstractMatrix)

solve `minarg_x |y-Ax|^2 + lambda sum_k |x_k|`
"""
function fit!(solver::LassoSolver, y::AbstractVector, A::AbstractMatrix, lambda::Real=1.0)
    x = similar(y, size(A,1))
    fit!(solver, x, y, A, lambda)
    return x
end

include("IRS.jl")
include("ISTA.jl")
include("FISTA.jl")
include("ADMM.jl")
