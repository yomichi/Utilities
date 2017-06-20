@compat mutable struct LassoFISTA <: LassoSolver
    tol::Float64
    maxiter::Int
    miniter::Int
    costs::Vector{Float64}
    function LassoFISTA(;tol::Real=1.0e-4, maxiter::Integer=1000, miniter::Integer=10)
        new(tol, maxiter, miniter, zeros(0))
    end
end

export LassoFISTA

function fit!(solver::LassoFISTA, x::Vector, y::AbstractVector, A::AbstractArray, lambda::Real=1.0)
    tol = solver.tol
    maxiter = solver.maxiter
    miniter = solver.miniter
    solver.costs = zeros(0)

    nx = length(x)
    ny = length(y)

    LinAlg.BLAS.gemv!('T',1.0,A,y,0.0,x)
    # x[:] = A'*y
    w = x[:]
    v = zeros(nx)

    Llambda = vecnorm(A'*A)
    invLlambda = 1.0/Llambda
    invL = invLlambda * lambda
    invlambda = 1.0/lambda
    beta = 0.0

    cost = 0.0
    LinAlg.BLAS.gemv!('N', 1.0, A, x, 0.0, v)
    @inbounds for i in 1:nx
        cost += 0.5*invlambda*(y[i] - v[i])^2
        cost += abs(x[i])
    end

    push!(solver.costs, cost)

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

        @inbounds for i in 1:nx
            w[i] = v[i] + r*(v[i]-x[i])
            x[i] = v[i]
        end
        LinAlg.BLAS.gemv!('N', 1.0, A, x, 0.0, v)
        cost = 0.0
        @inbounds for i in 1:nx
            cost += 0.5*invlambda*(y[i] - v[i])^2
            cost += abs(x[i])
        end
        push!(solver.costs, cost)

        if iter >= miniter && abs(cost - solver.costs[end-1])/cost < tol
            break
        end
    end

    return x
end
