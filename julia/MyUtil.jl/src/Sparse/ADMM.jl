export solve_L1_ADMM!, solve_L1_ADMM

function solve_L1_ADMM!(x::Vector, y::AbstractVector, A::AbstractMatrix, lambda::Real, mu::Real; tol::Real=1.0e-4, maxiter::Integer=1000)
    nx = length(x)
    ny = length(y)

    invmu = 1.0/mu

    ATy = A'*y
    z = similar(x)
    @inbounds for i in 1:nx
        x[i] = z[i] = ATy[i]
    end
    h = zeros(nx)

    # B = LinAlg.BLAS.gemm('T','N',1.0/lambda,A,A)
    # @inbounds for i in 1:nx
    #     B[i,i] += mu
    # end
    # cf = cholfact(Symmetric(B))

    B = mu*eye(nx) + (1.0/lambda)*A'*A

    invB = inv(B)

    for iter in 1:maxiter
        next_h = h + mu*(x-z)
        x[:] = invB*(ATy+mu*z-h)
        z[:] = soft_threshold(x-invmu*h, invmu)
        h[:] = next_h
    end

    #=
    while res > tol
        res = 0.0
        nrm = 0.0
        @inbounds for i in 1:nx
            x[i] = ATy[i] + mu*z[i] - h[i]
        end
        A_ldiv_B!(cf, x)
        @inbounds for i in 1:nx
            tmp = soft_threshold(x[i]-invmu*h[i], invmu)
            res += (tmp - z[i])^2
            nrm += tmp*tmp
            z[i] = tmp
            h[i] += mu*(x[i]-z[i])
        end
        res = sqrt((res/nrm)/nx)
        @show res
    end
    =#

    x[:] = z[:]

    return x
end

function solve_L1_ADMM(y::AbstractVector, A::AbstractMatrix, lambda::Real, mu::Real; tol::Real=1.0e-4)
    x = zeros(size(A,2))
    solve_L1_ADMM!(x,y,A,lambda,mu,tol=tol)
    return x
end
