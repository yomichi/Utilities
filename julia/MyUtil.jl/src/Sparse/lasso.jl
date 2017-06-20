export lasso!, lasso

"""
lasso!([solver_type,] x::Vector, y::Vector, A::AbstractMatrix)

solve `minarg_x |y-Ax|^2 + lambda sum_k |x_k|`
"""
lasso!(x::Vector, y::Vector, A::AbstractMatrix; opt...) = lasso!(LassoADMM, x, y, A; opt...)

"""
lasso([solver_type,] y::Vector, A::AbstractMatrix)

solve `minarg_x |y-Ax|^2 + lambda sum_k |x_k|`
"""
lasso(y::Vector, A::AbstractMatrix; opt...) = lasso(LassoADMM, y, A; opt...)

include("IRS.jl")
include("ISTA.jl")
include("FISTA.jl")
include("ADMM.jl")
