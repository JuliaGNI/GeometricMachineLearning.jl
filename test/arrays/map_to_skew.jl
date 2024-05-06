using GeometricMachineLearning: SkewSymMatrix, map_to_Skew 

function test_map_to_Skew(n::Int = 5)
    A = rand(SkewSymMatrix, n)
    @test A.S ≈ map_to_Skew(A)
end

test_map_to_Skew()