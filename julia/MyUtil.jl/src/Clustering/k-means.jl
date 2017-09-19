type KMeans{T}
    K :: Int
    N :: Int
    dim :: Int
    xs :: Vector{T}
    labels :: Vector{Int}
    centroids :: Vector{T}
    cost :: Float64
end

function KMeans{T}(xs::AbstractArray{T}, K::Integer)
    N = length(xs)
    dim = length(xs[1])
    labels = zeros(Int,N)
    centroids = shuffle(xs)[1:K]
    cost = 0.0
    km = KMeans(K, N, dim, xs, labels, centroids, cost)
    return update!(km)
end

function optimize!(km::KMeans; maxiter::Integer=1000)
    old_cost = km.cost
    for iter in 1:maxiter
        update!(km)
        if abs(cost-old_cost)/old_cost < rtol
            break
        end
        old_cost = cost
    end
    return km
end

function update!(km::KMeans)
    K = km.K
    N = km.N
    xs = km.xs
    dim = km.dim

    km.cost = 0.0
    ns = zeros(Int,K)
    cnext = [zeros(dim) for k in 1:K]
    for i in 1:N
        mn = Inf
        km.labels[i] = 0
        for k in 1:K
            d = sum((xs[i] .- km.centroids[k]).^2)
            if d < mn
                km.labels[i] = k
                mn = d
            end
        end
        km.cost += d
        ns[km.labels[i]] += 1
        cnext[km.labels[i]] .+= xs[i]
    end
    for k in 1:K
        if ns[k] > 0
            km.centroids[k] .= cnext[k] ./ ns[k]
        end
    end
    return km
end
