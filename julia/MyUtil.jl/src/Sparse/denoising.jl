export denoising!

function denoising!(solver, fs)
    N = length(fs)
    A = CosineFourierKernel(N,N)
    ks, result = fit_elbow!(solver, fs, A)
    return A*ks, ks, result
end

function denoising!(solver, fs, lambda)
    N = length(fs)
    A = CosineFourierKernel(N,N)
    ks = fit!(solver, fs, A, lambda)
    return A*ks, ks
end
