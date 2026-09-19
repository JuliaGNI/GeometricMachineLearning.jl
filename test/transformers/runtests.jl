using SafeTestsets

@safetestset "Test setup of transformer with Stiefel weights" begin
    include("transformer_setup.jl")
end
@safetestset "Sinusoidal positional encoding" begin
    include("positional_encoding.jl")
end
@safetestset "Check if the transformer can be applied to a tensor." begin
    include("transformer_application.jl")
end
@safetestset "Check if the gradient/pullback of MultiHeadAttention changes type in St case" begin
    include("transformer_gradient.jl")
end
@safetestset "Check if the optimization_step! changes the parameters of the transformer" begin
    include("transformer_optimizer.jl")
end
@safetestset "Regular transformer integrator" begin
    include("standard_transformer_integrator.jl")
end
@safetestset "Linear Symplectic Transformer" begin
    include("linear_symplectic_transformer.jl")
end
@safetestset "Symplectic Transformer chain construction" begin
    include("symplectic_transformer_chain.jl")
end
