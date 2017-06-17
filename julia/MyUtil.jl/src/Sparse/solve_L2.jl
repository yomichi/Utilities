export solve_L2!, solve_L2

"""
solve_L2!(x::Vector, y::Vector, A::Matrix, lambda::Real)
solve_L2!(x::Vector, y::Vector, A::Matrix, lambda::Real, etas::Vector)

solve `minarg_x |y-Ax|^2 + lambda sum_k x_k^2 / eta_k`
"""
function solve_L2!(x::Vector, y::Vector, A::Matrix, lambda::Real, etas::Vector)
    nk = length(etas)
    E = zeros(nk,nk)
    B = A'*A
    @inbounds for i in 1:nk
        B[i,i] += lambda/etas[i] 
    end
    cf = cholfact(B)
    LinAlg.BLAS.gemv!('T', 1.0, A, y, 0.0, x)
    return A_ldiv_B!(cf,x)
end

function solve_L2!(x::Vector, y::Vector, A::Matrix, lambda::Real)
    nk = size(A,2)
    return solve_L2(x,A,lambda, ones(nk))
end

"""
solve_L2(y::Vector, A::Matrix, lambda::Real)
solve_L2(y::Vector, A::Matrix, lambda::Real, etas::Vector)

solve `minarg_x |y-Ax|^2 + lambda sum_k x_k^2 / eta_k`
"""
function solve_L2(y::Vector, A::Matrix, lambda::Real, etas::Vector)
    x = similar(y)
    solve_L2!(x, y, A, lambda, etas)
    return x
end
function solve_L2(y::Vector, A::Matrix, lambda::Real)
    nk = size(A,2)
    return solve_L2(y,A,lambda, ones(nk))
end
