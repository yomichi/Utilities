export makesample, makesample!

function makesample(k::Integer, N::Integer, f=randn)
    x = f()
    xs = zeros(typeof(x), N)
    xs[1] = x
    for i in 2:k
        xs[i] = f()
    end
    shuffle!(xs)
    return xs
end

function makesample!(k::Integer, xs, f=randn)
    for i in 1:k
        xs[i] = f()
    end
    for i in (k+1):(length(xs))
        xs[i] = zero(xs[1])
    end
    shuffle!(xs)
    return xs
end
