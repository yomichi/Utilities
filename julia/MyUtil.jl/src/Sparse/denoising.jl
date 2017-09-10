export denoising!

function denoising!(solver, fs)
    N = length(fs)
    F = zeros(2N)
    F[1:N] = fs[:]
    F[(N+1):end] = fs[end:-1:1]
    A = CosineFourierKernel(2N,2N)
    ks, result, p = fit_elbow!(solver, F, A)
    LinAlg.BLAS.gemv!('N', 1.0, A, ks, 1.0, F)
    return F[1:N], ks, result
end

function denoising!(solver, fs, lambda)
    N = length(fs)
    F = zeros(2N)
    F[1:N] = fs[:]
    F[(N+1):end] = fs[end:-1:1]
    A = CosineFourierKernel(2N,2N)
    ks = fit!(solver, F, A, lambda)
    LinAlg.BLAS.gemv!('N', 1.0, A, ks, 1.0, F)
    return F[1:N], ks
end
