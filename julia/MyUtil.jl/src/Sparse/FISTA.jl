@compat immutable LassoFISTA end

export LassoFISTA

function lasso!(::Type{LassoFISTA}, x::Vector, y::AbstractVector, A::AbstractArray; lambda::Real=1.0, tol::Real=1.0e-4, miniter::Integer=10, maxiter::Integer=1000)
    nx = length(x)
    ny = length(y)

    LinAlg.BLAS.gemv!('T',1.0,A,y,0.0,x)
    # x[:] = A'*y
    w = x[:]
    v = zeros(nx)

    Llambda = vecnorm(A'*A)
    invLlambda = 1.0/Llambda
    invL = invLlambda * lambda
    beta = 0.0
    res = Inf

    for iter in 1:maxiter
        # v[:] = w + invLlambda*A'*(y-A*w)

        LinAlg.BLAS.gemv!('N',-1.0,A,w,0.0,v)
        @inbounds for i in 1:nx
            v[i] += y[i]
        end
        # w <- w + A'v/Llambda
        LinAlg.BLAS.gemv!('T',invLlambda,A,v,1.0,w)

        # v[:] = soft_threshold(v, invL)
        @inbounds for i in 1:nx
            v[i] = soft_threshold(w[i],invL)
        end

        next_beta = 0.5(1.0+sqrt(1.0+4.0*beta*beta))
        r = (beta-1.0)/next_beta
        beta = next_beta

        res = 0.0
        nrm = 0.0
        @inbounds for i in 1:nx
            w[i] = v[i] + r*(v[i]-x[i])
            res += (x[i]-v[i])^2
            x[i] = v[i]
            nrm += x[i]*x[i]
        end
        res = sqrt((res/nrm)/nx)

        if iter >= miniter && res < tol
            break
        end

    end

    return x
end

function lasso(::Type{LassoFISTA}, y::AbstractVector, A::AbstractArray; lambda::Real=1.0, tol::Real=1.0e-4, miniter::Integer=10, maxiter::Integer=1000)
    x = zeros(size(A,2))
    lasso!(LassoFISTA,x,y,A,lambda=lambda,tol=tol, miniter=miniter, maxiter=maxiter)
    return x
end
