export ridge!, ridge

"""
ridge!(x::Vector, y::Vector, A::Matrix, lambda::Real)
ridge!(x::Vector, y::Vector, A::Matrix, lambda::Real, etas::Vector)

solve `minarg_x |y-Ax|^2 + lambda sum_k x_k^2 / eta_k`
"""
function ridge!(x::Vector, y::Vector, A::Matrix, lambda::Real, etas::Vector)
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

function ridge!(x::Vector, y::Vector, A::Matrix, lambda::Real)
    nk = size(A,2)
    return ridge(x,A,lambda, ones(nk))
end

"""
ridge(y::Vector, A::Matrix, lambda::Real)
ridge(y::Vector, A::Matrix, lambda::Real, etas::Vector)

solve `minarg_x |y-Ax|^2 + lambda sum_k x_k^2 / eta_k`
"""
function ridge(y::Vector, A::Matrix, lambda::Real, etas::Vector)
    x = similar(y)
    ridge!(x, y, A, lambda, etas)
    return x
end
function ridge(y::Vector, A::Matrix, lambda::Real)
    nk = size(A,2)
    return ridge(y,A,lambda, ones(nk))
end
