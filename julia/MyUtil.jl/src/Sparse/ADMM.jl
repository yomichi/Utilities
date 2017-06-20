@compat immutable LassoADMM end

export LassoADMM

function lasso!(::Type{LassoADMM}, x::Vector, y::AbstractVector, A::AbstractMatrix; lambda::Real=1.0, mu::Real=1.0, tol::Real=1.0e-4, maxiter::Integer=1000)
    nx = length(x)
    ny = length(y)

    invmu = 1.0/mu

    ATy = LinAlg.BLAS.gemv('T', A, y)
    z = similar(x)
    @inbounds for i in 1:nx
        x[i] = z[i] = ATy[i]
    end
    h = zeros(nx)

    B = LinAlg.BLAS.gemm('T','N',1.0/lambda,A,A)
    @inbounds for i in 1:nx
        B[i,i] += mu
    end
    cf = cholfact(Symmetric(B))

    for iter in 1:maxiter
        res = 0.0
        nrm = 0.0
        @inbounds for i in 1:nx
            z[i] = ATy[i] + mu*x[i] - h[i]
        end
        A_ldiv_B!(cf, z)
        @inbounds for i in 1:nx
            tmp = soft_threshold(z[i]+invmu*h[i], invmu)
            res += (tmp - x[i])^2
            nrm += tmp*tmp
            x[i] = tmp
            h[i] += mu*(z[i]-x[i])
        end
        res = sqrt((res/nrm)/nx)
        if res < tol
            break
        end
    end

    return x
end

function lasso(::Type{LassoADMM}, y::AbstractVector, A::AbstractMatrix; lambda::Real=1.0, mu::Real=1.0, tol::Real=1.0e-4, maxiter::Integer=1000)
    x = zeros(size(A,2))
    lasso!(LassoADMM,x,y,A;lambda=lambda,mu=mu,tol=tol, maxiter=maxiter)
    return x
end
