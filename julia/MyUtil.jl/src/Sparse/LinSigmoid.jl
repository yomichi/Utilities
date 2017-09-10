module LinSigmoid

import LearnBase.fit
export linsigmoid

"""
linsigmoid(x::Real, p::Vector)

           x1
y1          -----
           /
          /
         /
y0 ------
        x0 

p = [x0, x1, y0, y1]
"""
linsigmoid(x::Real, p::Vector) = f(x,p)

function f(x::Real, p::Vector)
    if x < p[1]
        return p[3]
    elseif x < p[2]
        return ((p[4]-p[3])/(p[2]-p[1]))*(x-p[1])+p[3]
    else
        return p[4]
    end
end
function f(xs::Vector, p)
    return [f(x,p) for x in xs]
end

function grad_f(x::Real, p::Vector)
    g = zeros(4)
    if x < p[1]
        g[1] = 0.0
        g[2] = 0.0
        g[3] = 1.0
        g[4] = 0.0
    elseif x < p[2]
        a = (p[4]-p[3])/(p[2]-p[1])
        X = (x-p[1])/(p[2]-p[1])
        g[1] = a*(X-1.0)
        g[2] = -a*X
        g[3] = 1.0-X
        g[4] = X
    else
        g[1] = 0.0
        g[2] = 0.0
        g[3] = 0.0
        g[4] = 1.0
    end
    return g
end

function grad_cost(x::Real, y::Real, p)
    return (f(x,p)-y)*grad_f(x,p)
end
function grad_cost(xs::Vector, ys::Vector, p)
    g = zeros(4)
    for (x,y) in zip(xs,ys)
        g .+= grad_cost(x,y,p)
    end
    return g
end

function cost_f(xs::Vector, ys::Vector, p)
    return 0.5sum(abs2,ys.-f(xs,p))
end

function fit!(xs,ys,p; dt::Real=1.0e-4, tol::Real=1.0e-6, mu::Real=0.8)
    prev_dp = zeros(4)
    g = grad_cost(xs,ys,p)
    rho = sum(abs2,g)
    costs = [cost_f(xs,ys,p)]
    rc = Inf
    while !(0.0 <= rc < tol)
        dp = mu .* prev_dp .- dt.*g
        p .+= dp
        prev_dp .= dp
        g = grad_cost(xs,ys,p)
        rho = sum(abs2,g)
        push!(costs,cost_f(xs,ys,p))
        rc = (costs[end-1]-costs[end])/costs[end]
    end
    return p, costs
end

function fit(xs,ys, N::Integer=10; dt::Real=1.0e-4, tol::Real=1.0e-6, mu::Real=0.8)
    nx = length(xs)
    optp = zeros(4)
    optc = Inf
    for n in 1:N
        p = zeros(4)
        i1,i2 = rand(1:nx), rand(1:nx)
        p[1] = xs[i1]
        p[2] = xs[i2]
        while p[2] <= p[1]
            i1,i2 = rand(1:nx), rand(1:nx)
            p[1] = xs[i1]
            p[2] = xs[i2]
        end
        p[3] = ys[i1]
        p[4] = ys[i2]
        fit!(xs, ys, p, dt=dt, tol=tol, mu=mu)
        c = cost_f(xs, ys, p)
        if c < optc
            optp[:] = p[:]
            optc = c
        end
    end
    return optp
end

end
