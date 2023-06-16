using GeometricMachineLearning 
using CUDA
using Test
using GPUArrays  

gpu(A::AbstractArray) = GeometricMachineLearning.convert_to_gpu(CUDA.device(), A)

function test_apply_section(T, N, n)
    Y₁ = rand(StiefelManifold{T}, N, n)
    Y₂ = rand(StiefelManifold{T}, N, n)

    Y_gpu₁ = Y₁ |> gpu 
    Y_gpu₂ = Y₂ |> gpu 

    λY = GlobalSection(Y₁)
    
    λY_gpu = GlobalSection(Y_gpu₁)

    @time Y₃ = apply_section(λY, Y₂)

    @time Y_gpu₃ =  apply_section(λY_gpu, Y_gpu₂)

    @test (typeof(Y_gpu₃) <: StiefelManifold{T, AT} where {T, AT<:AbstractGPUMatrix})
end

T = Float32
for N = 1000:1000:5000
    n = N÷10
    print("N = ", N, " and n = ", n, "\n")
    test_apply_section(T, N, n)
    print("\n")
end