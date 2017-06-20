@compat immutable LassoIRS end

export LassoIRS
function lasso!(::Type{LassoIRS}, x::Vector, y::Vector, A::AbstractMatrix; lambda::Real=1.0, tol::Real=1.0e-4)
    nk = size(A,2)
    invnk = 1.0/nk
    etas = ones(nk)
    ft = zeros(nk)
    res = Inf
    while res > tol
        ridge!(x,y,A,lambda,etas)
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

function lasso(::Type{LassoIRS}, y::Vector, A::AbstractMatrix; lambda::Real=1.0, tol::Real=1.0e-4)
    x = similar(y)
    lasso!(LassoIRS, x,y,A,lambda,tol=tol)
    return x
end

