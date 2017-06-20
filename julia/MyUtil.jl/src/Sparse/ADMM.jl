@compat mutable struct LassoADMM <: LassoSolver
    mu::Float64
    tol::Float64
    maxiter::Int
    costs::Vector{Float64}
    function LassoADMM(;mu::Real=1.0, tol::Real=1.0e-4, maxiter::Integer=1000)
        new(mu, tol, maxiter, zeros(0))
    end
end

export LassoADMM

function fit!(solver::LassoADMM, x::Vector, y::AbstractVector, A::AbstractMatrix, lambda::Real=1.0)
    mu = solver.mu
    tol = solver.tol
    maxiter = solver.maxiter
    solver.costs = zeros(0)

    nx = length(x)
    ny = length(y)

    invmu = 1.0/mu
    invlambda = 1.0/lambda

    ATy = LinAlg.BLAS.gemv('T', A, y)
    z = similar(x)
    Ax = similar(x)
    @inbounds for i in 1:nx
        x[i] = z[i] = ATy[i]
    end
    h = zeros(nx)

    cost = 0.0
    LinAlg.BLAS.gemv!('N', 1.0, A, x, 0.0, Ax)
    @inbounds for i in 1:nx
        cost += 0.5*invlambda*(y[i] - Ax[i])^2
        cost += abs(x[i])
    end
    push!(solver.costs, cost)

    B = LinAlg.BLAS.gemm('T','N',invlambda,A,A)
    @inbounds for i in 1:nx
        B[i,i] += mu
    end
    cf = cholfact(Symmetric(B))

    for iter in 1:maxiter
        @inbounds for i in 1:nx
            x[i] = ATy[i] + mu*z[i] - h[i]
        end
        A_ldiv_B!(cf, x)
        @inbounds for i in 1:nx
            z[i] = soft_threshold(x[i]+invmu*h[i], invmu)
            h[i] += mu*(x[i]-z[i])
        end

        cost = 0.0
        LinAlg.BLAS.gemv!('N', 1.0, A, x, 0.0, Ax)
        @inbounds for i in 1:nx
            cost += 0.5*invlambda*(y[i] - Ax[i])^2
            cost += abs(x[i])
        end
        push!(solver.costs, cost)
        if abs(cost - solver.costs[end-1])/cost < tol
            break
        end
    end
    x[:] = z[:]
    return x
end

