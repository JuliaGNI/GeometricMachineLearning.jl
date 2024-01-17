using Test 
using GeometricMachineLearning: _compute_softmax 

function compute_softmax_test(T::DataType=Float32, n::Int=5, time_steps::Int=10)
    Z = rand(T, n, time_steps)
    A = rand(T, n, n)
    @test _compute_softmax(Z, A) ≈ softmax(Z' * A * Z)
end

compute_softmax_test()