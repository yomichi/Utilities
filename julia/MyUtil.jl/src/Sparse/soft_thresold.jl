export soft_threshold

soft_threshold(x::Real,lambda::Real) = sign(x) * max(zero(x), abs(x) - lambda)
soft_threshold(xs::AbstractArray, lambda::Real) = map(x->soft_threshold(x,lambda), xs)
