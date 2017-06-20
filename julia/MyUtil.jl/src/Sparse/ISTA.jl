@compat immutable LassoISTA end

export LassoISTA
function lasso!(::Type{LassoISTA}, x::Vector, y::AbstractVector, A::AbstractArray; lambda::Real=1.0, tol::Real=1.0e-4, maxiter::Integer=1000, miniter::Integer=10)
    nx = length(x)
    ny = length(y)

    LinAlg.BLAS.gemv!('T',1.0,A,y,0.0,x)
    w = zeros(nx)
    v = zeros(nx)

    Llambda = vecnorm(A'*A)
    invLlambda = 1.0/Llambda
    invL = invLlambda * lambda

    for iter in 1:maxiter
        # v <- y-Ax
        LinAlg.BLAS.gemv!('N',-1.0,A,x,0.0,v)
        @inbounds for i in 1:nx
            v[i] += y[i]
        end

        # w <- x + A'v/Llambda
        LinAlg.BLAS.gemv!('T',invLlambda,A,v,0.0,w)
        @inbounds for i in 1:nx
            w[i] += x[i]
        end

        # v <- next x
        @inbounds for i in 1:nx
            v[i] = soft_threshold(w[i],invL)
        end

        res = 0.0
        nrm = 0.0
        @inbounds for i in 1:nx
            res += (v[i]-x[i])^2
            nrm += v[i]*v[i]
            x[i] = v[i]
        end
        res = sqrt((res/nrm)/nx)
        if iter >= miniter && res < tol
            break
        end
    end
    return x
end

function lasso(::Type{LassoISTA}, y::AbstractVector, A::AbstractArray; lambda::Real=1.0, tol::Real=1.0e-4, maxiter::Integer=1000, miniter::Integer=10)
    x = zeros(size(A,2))
    lasso!(LassoISTA, x,y,A,lambda=lambda,tol=tol,maxiter=maxiter, miniter=miniter)
    return x
end
