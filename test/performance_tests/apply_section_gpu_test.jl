using GeometricMachineLearning 
using CUDA
using Test
using GPUArrays  
using Printf

function test_apply_section(dev, T, N, n)
    map_to_dev(A::AbstractArray) = GeometricMachineLearning.convert_to_dev(dev, A)

    Y₁ = rand(StiefelManifold{T}, N, n) |> map_to_dev
    Y₂ = rand(StiefelManifold{T}, N, n) |> map_to_dev

    λY = GlobalSection(Y₁)
    
    @time Y₃ = apply_section(λY, Y₂)

    if dev == CUDA.device()
        @test (typeof(Y₃) <: StiefelManifold{T, AT} where {T, AT<:AbstractGPUMatrix})
    end
end


T = Float32
for N = 1000:1000:5000
    n = N÷10
    print("N = ", N, " and n = ", n, "\n")
    @printf "GeometricMachineLearning cpu:  \n"
    test_apply_section(GeometricMachineLearning.CPUDevice(), T, N, n)
    @printf "GeometricMachineLearning gpu:  \n"
    test_apply_section(CUDA.device(), T, N, n)
    print("\n")
end