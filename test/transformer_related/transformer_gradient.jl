using Test, KernelAbstractions, GeometricMachineLearning, Zygote, LinearAlgebra
using GeometricMachineLearning: ResNetLayer
import Random 

Random.seed!(1234)

@doc raw"""
This checks if the gradients of the transformer change the type in case of the Stiefel manifold, and checks if they stay the same in the case of regular weights.
"""
function transformer_gradient_test(T, dim, n_heads, L, seq_length=8, batch_size=10)
    model₁ = Chain(Transformer(dim, n_heads, L, Stiefel=false), ResNetLayer(dim))
    model₂ = Chain(Transformer(dim, n_heads, L, Stiefel=true), ResNetLayer(dim))

    ps₁ = NeuralNetwork(model₁, KernelAbstractions.CPU(), T).params
    ps₂ = NeuralNetwork(model₂, KernelAbstractions.CPU(), T).params
    
    input₁ = rand(T, dim, seq_length, batch_size)
    input₂ = rand(T, dim, seq_length, batch_size)
    
    loss₁(ps, input) = norm(model₁(input, ps))
    loss₂(ps, input) = norm(model₂(input, ps))
    grad₁ = Zygote.gradient(ps -> loss₁(ps, input₁), ps₁)[1]
    grad₂ = Zygote.gradient(ps -> loss₁(ps, input₂), ps₁)[1]
    grad₃ = Zygote.gradient(ps -> loss₂(ps, input₁), ps₂)[1]
    grad₄ = Zygote.gradient(ps -> loss₂(ps, input₂), ps₂)[1]

    @test typeof(NeuralNetworkParameters(grad₁.params)) == typeof(ps₁)
    @test typeof(NeuralNetworkParameters(grad₂.params)) == typeof(ps₁)
    @test typeof(NeuralNetworkParameters(grad₃.params)) != typeof(ps₂)
    @test typeof(NeuralNetworkParameters(grad₄.params)) != typeof(ps₂)
end

transformer_gradient_test(Float32, 10, 5, 4)