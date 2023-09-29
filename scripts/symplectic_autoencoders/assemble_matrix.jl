using OffsetArrays

function assemble_matrix(μ::T, Δx::T, N=2048) where T 
    K = zeros(T, N+2, N+2)
    K = OffsetArray(K, OffsetArrays.Origin(0,0))
    fac = μ^2/(2*Δx)
    for i in 1:N
        K[i,i] += fac 
        K[i,i+1] -= fac 
        K[i,i-1] -= fac 
        K[i-1,i-1] += fac/T(2)
        K[i+1,i+1] += fac/T(2)
    end
    K 
end

function assemble_matrix_old(μ::T, Δx::T, N=2048) where T
    K = zeros(T, N+2, N+2)
    K = OffsetArray(K, OffsetArrays.Origin(0,0))
    fac = μ^2/Δx
    for ij in zip((0,N+1), (0,N+1))
        K[ij...] = fac/T(4)
    end
    for ij in zip((1,N), (0,N+1))
        K[ij...] = -fac/T(2)
    end
    K[1,1] = T(3/4)*fac 
    K[N,N] = T(3/4)*fac
    K[1,2] = -fac/T(2)
    K[2,1] = -fac/T(2)
    for i in 2:(N-1)
        K[i,i] = fac
        for ij0 in zip((i, i+1), (i+1,i))
            K[ij0...] = -fac/T(2)
        end
    end
    K
end