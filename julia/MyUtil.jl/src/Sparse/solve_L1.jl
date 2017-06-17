export solve_L1!, solve_L1

"""
solve_L1!(x::Vector, y::Vector, A::AbstractMatrix, lambda::Real; tol::Real=1.0e-4)

solve `minarg_x |y-Ax|^2 + lambda sum_k |x_k|`
"""
function solve_L1!(x::Vector, y::Vector, A::AbstractMatrix, lambda::Real; tol::Real=1.0e-4)
    nk = size(A,2)
    invnk = 1.0/nk
    etas = ones(nk)
    ft = zeros(nk)
    res = Inf
    while res > tol
        solve_L2!(x,y,A,lambda,etas)
        nrm = 0.0
        res = 0.0
        @inbounds for i in 1:nk
            r = abs(x[i]) - etas[i]
            etas[i] += r
            res += r^2
            nrm += etas[i]^2
        end
        res /=  nrm
        res *= invnk
    end
    return x
end

"""
solve_L1(y::Vector, A::AbstractMatrix, lambda::Real; tol::Real=1.0e-4)

solve `minarg_x |y-Ax|^2 + lambda sum_k |x_k|`
"""
function solve_L1(y::Vector, A::AbstractMatrix, lambda::Real; tol::Real=1.0e-4)
    x = similar(y)
    solve_L1!(x,y,A,lambda,tol=tol)
    return x
end

