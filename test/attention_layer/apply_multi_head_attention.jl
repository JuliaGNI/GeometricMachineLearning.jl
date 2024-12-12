using GeometricMachineLearning, Test 
using NNlib: softmax
using LinearAlgebra: I
import Random 

Random.seed!(1234)

function compare_attention_to_mha(N, batch_size=10, T=Float32)
    model₁ = MultiHeadAttention(N, 1, add_connection=false)
    model₂ = Attention(N, softmax, add_connection=false)
    model₃ = MultiHeadAttention(N, 1, add_connection=true)
    model₄ = Attention(N, softmax, add_connection=true)

    ps₂ = NeuralNetwork(model₂, CPU(), T).params
    ps₁ = (PQ=(head_1=ps₂.PQ,), PK=(head_1=ps₂.PK,), PV=(head_1=typeof(ps₂.PK)(I(N)),))

    mat = rand(T, N, N)
    ten = rand(T, N, N, batch_size)
    @test isapprox(model₁(mat, ps₁), model₂(mat, ps₂))
    @test isapprox(model₁(ten, ps₁), model₂(ten, ps₂))

    @test isapprox(model₃(mat, ps₁), model₄(mat, ps₂))
    @test isapprox(model₃(ten, ps₁), model₄(ten, ps₂))
end

compare_attention_to_mha(10)