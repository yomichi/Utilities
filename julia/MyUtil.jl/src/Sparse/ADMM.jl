type LassoADMM <: LassoSolver
    mu::Float64
    tol::Float64
    maxiter::Int
    miniter::Int
    costs::Vector{Float64}
    function LassoADMM(;mu::Real=1.0, tol::Real=1.0e-4, maxiter::Integer=1000, miniter::Integer=2)
        new(mu, tol, maxiter, miniter, zeros(0))
    end
end

type LassoResult
    optlambda::Float64
    optnonzero::Int
    lambdas::Vector{Float64}
    residues::Vector{Float64}
    penalties::Vector{Float64}
    costs::Vector{Float64}
    nonzeros::Vector{Int}
end

export LassoADMM, LassoResult
export fit_elbow!

function fit!(solver::LassoADMM, x::Vector, y::AbstractVector, A::AbstractMatrix, lambda::Real=1.0)
    nx = length(x)
    B = A'*A
    @inbounds for i in 1:nx
        B[i,i] += solver.mu
    end
    cf = cholfact(Symmetric(B))
    return fit_impl!(solver, x, y, A, cf, lambda)
end

function fit_elbow!(solver::LassoADMM, y::AbstractVector, A::AbstractMatrix)
    x = zeros(size(A,2))
    return fit_elbow!(solver, x, y, A)
end

function fit_elbow!(solver::LassoADMM, x::Vector, y::AbstractVector, A::AbstractMatrix)
    nx = length(x)
    B = LinAlg.BLAS.gemm('T','N',1.0,A,A)
    @inbounds for i in 1:nx
        B[i,i] += solver.mu
    end
    cf = cholfact(Symmetric(B))

    N = length(y)
    K = length(x)
    invK = 1.0/K

    Ax = zeros(N)

    lambda = 1.0
    fit_impl!(solver, x, y, A, cf, lambda)
    nonzero, res, penalty, cost = lasso_cost!(x,y,A,Ax,lambda)
    lambdas = [lambda]
    residues = [res]
    penalties = [penalty]
    costs = [cost]
    nonzeros = [nonzero]

    while nonzero > 0 && lambda < 1.0e8
        lambda *= 2.0
        fit_impl!(solver, x, y, A, cf, lambda)
        nonzero, res, penalty, cost = lasso_cost!(x,y,A,Ax,lambda)
        push!(lambdas, lambda)
        push!(residues, res)
        push!(penalties, penalty)
        push!(costs, cost)
        push!(nonzeros, nonzero)
    end

    high_lambda = lambdas[end]
    high_res = residues[end]
    lambda = lambdas[1]

    while nonzero < K && lambda > 1.0e-8
        lambda *= 0.5
        fit_impl!(solver, x, y, A, cf, lambda)
        nonzero, res, penalty, cost = lasso_cost!(x,y,A,Ax,lambda)
        if(nonzero == 0)
            high_lambda = lambdas[end] = lambda
            high_res = residues[end] = res
            penalties[end] = penalty
            costs[end] = cost
            nonzeros[end] = nonzero
        else
            push!(lambdas, lambda)
            push!(residues, res)
            push!(penalties, penalty)
            push!(costs, cost)
            push!(nonzeros, nonzero)
        end
    end

    #=
    high_lambda = log(high_lambda)
    low_lambda = log(lambdas[end])
    low_res = residues[end]
    slope = (high_res-low_res)/(high_lambda-low_lambda)

    f = x->low_res + slope*(log(x)-low_lambda)

    max_diff = 0.0
    imax = 1
    for i in 1:length(lambdas)
        res = f(lambdas[i])
        diff = res - residues[i]
        if diff > max_diff
            max_diff = diff
            imax = i
        end
    end
    optlambda = lambdas[imax]
    optnonzero = nonzeros[imax]
    =#

    p = fit(log.(lambdas), residues,10, dt=1.0e-6, mu=0.9)
    @show p
    optlambda = exp(p[1])
    fit_impl!(solver, x, y, A, cf, optlambda)

    nonzero, res, penalty, cost = lasso_cost!(x,y,A,Ax,lambda)

    push!(lambdas, optlambda)
    push!(residues, res)
    push!(penalties, penalty)
    push!(costs, cost)
    push!(nonzeros, nonzero)

    sortedidx = sortperm(lambdas)

    result = LassoResult(optlambda, nonzero, lambdas[sortedidx], residues[sortedidx], penalties[sortedidx], costs[sortedidx], nonzeros[sortedidx])

    return x, result, p
end

function fit_impl!(solver::LassoADMM, x::Vector, y::AbstractVector, A::AbstractMatrix, cf::LinAlg.Cholesky, lambda::Real)
    @show lambda
    mu = solver.mu
    tol = solver.tol
    maxiter = solver.maxiter
    solver.costs = zeros(0)
    stcoeff = lambda/solver.mu

    nx = length(x)
    ny = length(y)

    invnx = 1.0/nx
    invmu = 1.0/mu

    ATy = A'*y
    z = zeros(nx)
    Ax = zeros(ny)
    @inbounds for i in 1:nx
        x[i] = z[i] = ATy[i]
    end
    h = zeros(nx)

    cost = 0.0
    LinAlg.BLAS.gemv!('N', 1.0, A, x, 0.0, Ax)
    @inbounds for i in 1:ny
        cost += 0.5*(y[i] - Ax[i])^2
    end
    @inbounds for i in 1:nx
        cost += lambda*abs(x[i])
    end
    cost *= invnx
    old_cost = cost

    for iter in 1:maxiter
        @inbounds for i in 1:nx
            x[i] = ATy[i] + mu*z[i] - h[i]
        end
        A_ldiv_B!(cf, x)
        @inbounds for i in 1:nx
            z[i] = soft_threshold(x[i]+invmu*h[i], stcoeff)
            h[i] += mu*(x[i]-z[i])
        end

        cost = 0.0
        LinAlg.BLAS.gemv!('N', 1.0, A, x, 0.0, Ax)
        @inbounds for i in 1:ny
            cost += 0.5*(y[i] - Ax[i])^2
        end
        @inbounds for i in 1:nx
            cost += lambda*abs(x[i])
        end
        if isinf(cost)
            error("cost diverges!")
        end
        cost *= invnx
        push!(solver.costs, cost)
        if abs(cost - old_cost)/cost < tol && iter >= solver.miniter
            break
        end
        old_cost = cost
    end
    x[:] = z[:]
    return x
end


