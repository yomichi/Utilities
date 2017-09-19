export KMeans, optimize!

type KMeans{T}
    K :: Int
    N :: Int
    dim :: Int
    xs :: Matrix{T}
    labels :: Vector{Int}
    centroids :: Matrix{T}
    cost :: Float64
end

function KMeans{T<:Real}(xs::AbstractMatrix{T}, K::Integer)
    dim = size(xs,1)
    N = size(xs,2)
    labels = rand(1:K,N)
    labels[1:K] .= collect(1:K)
    shuffle!(labels)
    centroids = zeros(T,dim,K)
    ns = zeros(Int,K)
    for i in 1:N
        k = labels[i]
        centroids[:,k] .+= xs[:,i]
        ns[k] += 1
    end
    for k in 1:K
        centroids[:,k] ./= ns[k]
    end
    cost = 0.0
    return KMeans(K, N, dim, xs, labels, centroids, cost)
end
function KMeans{T<:Real}(xs::AbstractVector{T}, K::Integer)
    KMeans(reshape(xs,1,:), K)
end

function optimize!(km::KMeans; maxiter::Integer=1000, rtol::Real=1.0e-4)
    old_cost = km.cost
    for iter in 1:maxiter
        update!(km)
        if abs(km.cost-old_cost)/old_cost < rtol
            break
        end
        old_cost = km.cost
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
    cnext = zeros(dim,K)
    for i in 1:N
        mn = Inf
        km.labels[i] = 0
        d = 0.0
        for k in 1:K
            d = sum((xs[:,i] .- km.centroids[:,k]).^2)
            if d < mn
                km.labels[i] = k
                mn = d
            end
        end
        km.cost += d
        ns[km.labels[i]] += 1
        cnext[:,km.labels[i]] .+= xs[:,i]
    end
    for k in 1:K
        if ns[k] > 0
            km.centroids[:,k] .= cnext[:,k] ./ ns[k]
        end
    end
    return km
end
