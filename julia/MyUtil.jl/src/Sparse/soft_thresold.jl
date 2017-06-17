export soft_threshold

soft_threshold(x::Real,lambda::Real) = ifelse(x>lambda, x-lambda, ifelse(x<-lambda, x+lambda, 0.0))
soft_threshold(xs::AbstractArray, lambda::Real) = map(x->soft_threshold(x,lambda), xs)
