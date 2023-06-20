using GeometricMachineLearning, CUDA


function update_cache_test(dev, N, n)
    map_to_dev(A::AbstractArray) = GeometricMachineLearning.convert_to_dev(CUDA.device(), A)

    B₁ = rand(StiefelLieAlgHorMatrix{T}, N, n) |> gpu 
    B₂ = rand(StiefelLieAlgHorMatrix{T}, N, n) |> gpu

    cache = AdamCache(B₁, B₂)

end