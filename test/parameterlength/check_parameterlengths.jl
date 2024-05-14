using Test, GeometricMachineLearning
import Random 

Random.seed!(1234)

function check_parameterlength(N, n)
    model₁ = Dense(N,N,identity)
    model₂ = Dense(N,N,identity,use_bias=false)
    model₃ = StiefelLayer(N,N)
    model₄ = StiefelLayer(N,n)
    model₃₂ = Chain(model₃, model₂)
    model₅ = MultiHeadAttention(N, N÷n, Stiefel=false)
    model₆ = MultiHeadAttention(N, N÷n, Stiefel=true)
    model₇ = VolumePreservingAttention(N, skew_sym = true)
    model₈ = VolumePreservingAttention(N, skew_sym = false)

    @test parameterlength(model₁) == N*N+N
    @test parameterlength(model₂) == N*N 
    @test parameterlength(model₃) == Int(N*(N-(N+1)/2))
    @test parameterlength(model₄) == Int(n*(N-(n+1)/2))
    @test parameterlength(model₃₂) == parameterlength(model₃) + parameterlength(model₂)
    @test parameterlength(model₅) == 3*N*N
    @test parameterlength(model₆) == 3*N÷n*Int(n*(N-(n+1)/2))
    @test parameterlength(model₇) == N * (N - 1) ÷ 2
    @test parameterlength(model₈) == N * N
end

check_parameterlength(10,5)