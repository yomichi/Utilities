function solve_L1_FISTA!(x::Vector, y::AbstractVector, A::AbstractArray; tol::Real=1.0e-4)
    nx = length(x)
    ny = length(y)

    fill!(x,0.0)
    w = zeros(nx)
    v = zeros(nx)

    Llambda = vecnorm(A'*A)
    invLlambda = 1.0/Llambda
    invL = invLlambda * lambda
    beta = 1.0
    res = Inf

    while res > tol
        # v = w+A'*(y-A*w)*invLlambda
        LinAlg.BLAS.gemv!('N',-1.0,A,w,0.0,v)
        @inbounds for i in 1:nx
            v[i] += y[i]
        end
        LinAlg.BLAS.gemv!('T',invLlambda,A,v,1.0,w)
        @inbounds for i in 1:nx
            v[i] = soft_threshold(w[i],invL)
        end
        next_beta = 0.5(1.0+sqrt(1.0+4.0*beta*beta))
        r = (beta-1.0)/next_beta
        beta = next_beta
        res = 0.0
        nrm = 0.0
        @inbounds for i in 1:nx
            w[i] = x[i] + r*(v[i]-x[i])
            res += (x[i]-v[i])^2
            x[i] = v[i]
            nrm += x[i]*x[i]
        end
        res = sqrt((res/nrm)/nk)
    end
    return x
end

function solve_L1_FISTA(y::AbstractVector, A::AbstractArray; tol::Real=1.0e-4)
    x = zeros(size(A,2))
    solve_L1_FISTA!(x,y,A,tol=tol)
    return x
end
