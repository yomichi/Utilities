function solve_L1_ADMM!(x::Vector, y::AbstractVector, A::AbstractMatrix, lambda::Real, mu::Real; tol::Real=1.0e-4)
    nx = length(x)
    ny = length(y)

    invmu = 1.0/mu

    ATy = A'*y
    z = similar(x)
    @inbounds for i in 1:nx
        x[i] = z[i] = ATy[i]
    end
    h = zeros(nx)

    B = LinAlg.BLAS.gemm('T','N',1.0/lambda,A,A)
    @inbounds for i in 1:nx
        B[i,i] += mu
    end
    cf = cholfact(B)

    res = Inf
    while res > tol
        res = 0.0
        nrm = 0.0
        @inbounds for i in 1:nx
            z[i] = ATy[i] + mu*x[i] - h[i]
        end
        A_ldiv_B!(cf, z)
        @inbounds for i in 1:nx
            tmp = soft_threshold(z[i]-invmu*h[i], invmu)
            res += (tmp - x[i])^2
            nrm += tmp*tmp
            x[i] = tmp
            h[i] += mu*(z[i]-x[i])
        end
        res = sqrt((res/nrm)/nx)
    end
    return x
end

function solve_L1_ADMM(y::AbstractVector, A::AbstractMatrix, lambda::Real, mu::Real; tol::Real=1.0e-4)
    x = zeros(size(A,2))
    solve_L1_ADMM!(x,y,A,lambda,mu,tol=tol)
    return x
end
