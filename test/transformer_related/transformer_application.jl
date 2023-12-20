using Test, KernelAbstractions, GeometricMachineLearning

@doc raw"""
This tests if the size of the input array is kept constant when fed into the transformer (for a matrix and a tensor)
"""
function transformer_application_test(T, dim, n_heads, L, seq_length=8, batch_size=10)
    model₁ = Chain(Transformer(dim, n_heads, L, Stiefel=false), ResNet(dim))
    model₂ = Chain(Transformer(dim, n_heads, L, Stiefel=true), ResNet(dim))

    ps₁ = initialparameters(KernelAbstractions.CPU(), T, model₁)
    ps₂ = initialparameters(KernelAbstractions.CPU(), T, model₂)
    
    input₁ = rand(T, dim, seq_length, batch_size)
    input₂ = rand(T, dim, seq_length)

    @test size(model₁(input₁, ps₁)) == size(input₁)
    @test size(model₂(input₁, ps₂)) == size(input₁)
    @test size(model₁(input₂, ps₁)) == size(input₂)
    @test size(model₂(input₂, ps₂)) == size(input₂)
end

transformer_application_test(Float32, 10, 5, 4)