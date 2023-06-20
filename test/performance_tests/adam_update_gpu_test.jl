using GeometricMachineLearning, CUDA


function adam_update_test(dev, N, n)
    map_to_dev(A::AbstractArray) = GeometricMachineLearning.convert_to_dev(CUDA.device(), A)

    B₁ = rand(StiefelLieAlgHorMatrix{T}, N, n) |> map_to_dev
    B₂ = rand(StiefelLieAlgHorMatrix{T}, N, n) |> map_to_dev

    cache = AdamCache(B₁, B₂)

    o 
end