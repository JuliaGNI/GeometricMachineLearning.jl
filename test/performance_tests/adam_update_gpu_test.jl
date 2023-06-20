using GeometricMachineLearning, CUDA
using Test, Printf
using GPUArrays

function adam_update_test(dev, T, N, n)
    map_to_dev(A::AbstractArray) = GeometricMachineLearning.convert_to_dev(dev, A)

    B₁ = rand(StiefelLieAlgHorMatrix{T}, N, n) |> map_to_dev
    B₂ = rand(StiefelLieAlgHorMatrix{T}, N, n) |> map_to_dev
    cache = AdamCache(B₁, B₂)

    B₃ = rand(StiefelLieAlgHorMatrix{T}, N, n) |> map_to_dev

    o = AdamOptimizer()

    @time update!(o, cache, B₃)

    if dev == CUDA.device()
        @test (typeof(cache.B₁) <: GeometricMachineLearning.StiefelLieAlgHorMatrix{T, GeometricMachineLearning.SkewSymMatrix{T, VT}, AT} where {T, VT <: AbstractGPUVector{T}, AT <: AbstractGPUMatrix{T}})
        @test (typeof(cache.B₂) <: GeometricMachineLearning.StiefelLieAlgHorMatrix{T, GeometricMachineLearning.SkewSymMatrix{T, VT}, AT} where {T, VT <: AbstractGPUVector{T}, AT <: AbstractGPUMatrix{T}})
        @test (typeof(B₃)       <: GeometricMachineLearning.StiefelLieAlgHorMatrix{T, GeometricMachineLearning.SkewSymMatrix{T, VT}, AT} where {T, VT <: AbstractGPUVector{T}, AT <: AbstractGPUMatrix{T}})
    end
end

T = Float32
for N = 1000:1000:10000
    n = N÷10
    print("N = ", N, " and n = ", n, "\n")
    @printf "GeometricMachineLearning cpu:  " 
    adam_update_test(GeometricMachineLearning.CPUDevice(), T, N, n)
    @printf "GeometricMachineLearning gpu:  " 
    adam_update_test(CUDA.device(), T, N, n)
    print("\n")
end
