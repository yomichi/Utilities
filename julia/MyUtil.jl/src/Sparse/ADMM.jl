type LassoADMM <: LassoSolver
    mu::Float64
    tol::Float64
    maxiter::Int
    costs::Vector{Float64}
    function LassoADMM(;mu::Real=1.0, tol::Real=1.0e-4, maxiter::Integer=1000)
        new(mu, tol, maxiter, zeros(0))
    end
end

type LassoResult
    optlambda::Float64
    lambdas::Vector{Float64}
    residues::Vector{Float64}
    nonzeros::Vector{Int}
end

export LassoADMM, LassoResult
export fit_elbow, fit_elbow!

function fit!(solver::LassoADMM, x::Vector, y::AbstractVector, A::AbstractMatrix, lambda::Real=1.0)
    nx = length(x)
    B = LinAlg.BLAS.gemm('T','N',1.0,A,A)
    @inbounds for i in 1:nx
        B[i,i] += solver.mu
    end
    cf = cholfact(Symmetric(B))
    return fit_impl!(solver, x, y, A, cf, lambda)
end

function fit_elbow(solver::LassoADMM, y::AbstractVector, A::AbstractMatrix)
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
    res = 0.0
    nonzero = 0

    fit_impl!(solver, x, y, A, cf, lambda)
    LinAlg.BLAS.gemv!('N', 1.0, A, x, 0.0, Ax)
    @inbounds for i in 1:N
        res += (y[i]-Ax[i])^2
    end
    @inbounds for i in 1:K
        nonzero += ifelse(x[i]==0.0, 0.0, 1)
    end

    lambdas = [lambda]
    residues = [res]
    nonzeros = [nonzero]

    while nonzero > 0 && lambda < 1.0e8
        lambda *= 2.0
        fit_impl!(solver, x, y, A, cf, lambda)
        LinAlg.BLAS.gemv!('N', 1.0, A, x, 0.0, Ax)
        res = 0.0
        nonzero = 0
        @inbounds for i in 1:N
            res += (y[i]-Ax[i])^2
        end
        @inbounds for i in 1:K
            nonzero += ifelse(x[i]==0.0, 0.0, 1)
        end
        push!(lambdas, lambda)
        push!(residues, res)
        push!(nonzeros, nonzero)
    end

    high_lambda = lambdas[end]
    high_res = residues[end]
    lambda = lambdas[1]

    while nonzero < K && lambda > 1.0e-8
        lambda *= 0.5
        fit_impl!(solver, x, y, A, cf, lambda)
        LinAlg.BLAS.gemv!('N', 1.0, A, x, 0.0, Ax)
        res = 0.0
        nonzero = 0
        @inbounds for i in 1:N
            res += (y[i]-Ax[i])^2
        end
        @inbounds for i in 1:K
            nonzero += ifelse(x[i]==0.0, 0.0, 1)
        end
        if(nonzero == 0)
            high_lambda = lambdas[end] = lambda
            high_res = residues[end] = res
            nonzeros[end] = nonzero
        else
            push!(lambdas, lambda)
            push!(residues, res)
            push!(nonzeros, nonzero)
        end
    end
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
    sortedidx = sortperm(lambdas)

    result = LassoResult(optlambda, lambdas[sortedidx], residues[sortedidx], nonzeros[sortedidx])

    return fit_impl!(solver, x, y, A, cf, optlambda), result
end

function fit_impl!(solver::LassoADMM, x::Vector, y::AbstractVector, A::AbstractMatrix, cf::LinAlg.Cholesky, lambda::Real)
    mu = solver.mu
    tol = solver.tol
    maxiter = solver.maxiter
    solver.costs = zeros(0)
    stcoeff = lambda/solver.mu

    nx = length(x)
    ny = length(y)

    invnx = 1.0/nx
    invmu = 1.0/mu

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
        cost += 0.5*(y[i] - Ax[i])^2
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
        @inbounds for i in 1:nx
            cost += 0.5*(y[i] - Ax[i])^2
            cost += lambda*abs(x[i])
        end
        cost *= invnx
        push!(solver.costs, cost)
        if abs(cost - old_cost)/cost < tol
            break
        end
    end
    x[:] = z[:]
    return x
end


