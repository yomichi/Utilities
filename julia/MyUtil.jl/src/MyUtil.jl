__precompile__(true)

module MyUtil

using Compat
using LearnBase
using LsqFit

include("Sparse/sparse.jl")
include("Clustering/k-means.jl")

end
