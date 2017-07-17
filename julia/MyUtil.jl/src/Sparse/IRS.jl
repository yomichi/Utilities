type LassoIRS <: LassoSolver
    tol::Float64
    maxiter::Int
    costs::Vector{Float64}
    function LassoIRS(;tol::Float64=1.0e-4, maxiter::Integer=1000)
        new(tol, maxiter, zeros(0))
    end
end
export LassoIRS

function fit!(solver::LassoIRS, x::Vector, y::Vector, A::AbstractMatrix, lambda::Real=1.0)
    tol = solver.tol
    maxiter = solver.maxiter
    solver.costs = zeros(0)
    nx = size(A,1)
    nk = size(A,2)
    Ax = zeros(nx)
    invnk = 1.0/nk
    invlambda = 1.0/lambda
    etas = ones(nk)
    ft = zeros(nk)

    cost = 0.0
    @inbounds for i in 1:nx
        cost += 0.5*invlambda*y[i]^2
    end
    push!(solver.costs, cost)

    for iter in 1:maxiter
        ridge!(x,y,A,lambda,etas)
        nrm = 0.0
        res = 0.0
        @inbounds for i in 1:nk
            r = abs(x[i]) - etas[i]
            etas[i] += r
            res += r^2
            nrm += etas[i]^2
        end
        LinAlg.BLAS.gemv!('N', 1.0, A, x, 0.0, Ax)
        cost = 0.0
        @inbounds for i in 1:nx
            cost += 0.5*invlambda*(y[i]-Ax[i])^2
            cost += abs(x[i])
        end
        push!(solver.costs, cost)
        if abs(cost - solver.costs[end-1])/cost < tol
            break
        end
    end
    return x
end

