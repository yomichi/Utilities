@compat mutable struct LassoISTA <: LassoSolver
    tol::Float64
    maxiter::Int
    miniter::Int
    costs::Vector{Float64}
    function LassoISTA(;tol::Real=1.0e-4, maxiter::Integer=1000, miniter::Integer=10)
        new(tol, maxiter, miniter, zeros(0))
    end
end
export LassoISTA

function fit!(solver::LassoISTA, x::Vector, y::AbstractVector, A::AbstractArray, lambda::Real=1.0)
    tol = solver.tol
    maxiter = solver.maxiter
    miniter = solver.miniter
    solver.costs = zeros(0)
    nx = length(x)
    ny = length(y)

    LinAlg.BLAS.gemv!('T',1.0,A,y,0.0,x)
    w = zeros(nx)
    v = zeros(nx)

    Llambda = vecnorm(A'*A)
    invLlambda = 1.0/Llambda
    invL = invLlambda * lambda
    invlambda = 1.0/lambda

    # v <- y-Ax
    LinAlg.BLAS.gemv!('N',1.0,A,x,0.0,v)
    cost = 0.0
    @inbounds for i in 1:nx
        v[i] = y[i] - v[i]
        cost += 0.5*invlambda*v[i]*v[i]
    end
    push!(solver.costs, cost)

    for iter in 1:maxiter
        # w <- x + A'v/Llambda
        LinAlg.BLAS.gemv!('T',invLlambda,A,v,0.0,w)
        @inbounds for i in 1:nx
            w[i] += x[i]
        end

        @inbounds for i in 1:nx
            x[i] = soft_threshold(w[i],invL)
        end

        # v <- y-Ax
        LinAlg.BLAS.gemv!('N',1.0,A,x,0.0,v)
        @inbounds for i in 1:nx
            v[i] = y[i] - v[i]
        end

        cost = 0.0
        @inbounds for i in 1:nx
            cost += 0.5*invlambda*v[i]^2
            cost += abs(x[i])
        end
        push!(solver.costs, cost)
        if iter >= miniter && abs(cost - solver.costs[end-1])/cost < tol
            break
        end

    end
    return x
end

