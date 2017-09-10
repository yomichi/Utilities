export LassoSolver
@compat abstract type LassoSolver end

include("LinSigmoid.jl")
using .LinSigmoid
export LinSigmoid

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

"""
lasso_cost(x, y, A, lambda)
lasso_cost!(x, y, A, Ax, lambda)

return `nonzero`, `res`, `penalty`, `cost`
    `nonzero = count(x==0.0,x)`
    `res = 0.5sum(abs2,y-A*x)`
    `penalty = sum(abs,x)`
    `cost = res + lambda*penalty`

    `Ax` in lasso_cost! will be replace by A*x
"""
function lasso_cost!(x, y, A, Ax, lambda)
    res = 0.0
    nonzero = 0.0
    penalty = 0.0

    LinAlg.BLAS.gemv!('N', 1.0, A, x, 0.0, Ax)
    @inbounds for i in 1:(length(y))
        res += 0.5(y[i]-Ax[i])^2
    end
    @inbounds for i in 1:(length(x))
        nonzero += ifelse(x[i]==0.0, 0.0, 1.0)
        penalty += abs(x[i])
    end

    return nonzero, res, penalty, res+lambda*penalty
end

function lasso_cost(x, y, A, lambda)
    Ax = zeros(y)
    return lasso_cost!(x,y,A,Ax,lambda)
end

include("IRS.jl")
include("ISTA.jl")
include("FISTA.jl")
include("ADMM.jl")
