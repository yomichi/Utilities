export CosineFourierKernel

function CosineFourierKernel(nx::Integer, ny::Integer)
    coeff = sqrt(2/nx)
    A = zeros(nx,ny)
    for y in 0:(ny-1)
        for x in 0:(nx-1)
            A[x+1,y+1] = coeff*cospi((x+0.5)*(y+0.5)/nx)
        end
    end
    return A
end
