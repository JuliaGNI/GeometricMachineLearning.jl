using SafeTestsets

@safetestset "Attention layer #1" begin
    include("attention_setup.jl")
end
@safetestset "Test setup of MultiHeadAttention layer Stiefel weights" begin
    include("multi_head_attention_stiefel_setup.jl")
end
@safetestset "Test geodesic and Cayley retr for the MultiHeadAttention layer w/ St weights" begin
    include("multi_head_attention_stiefel_retraction.jl")
end
@safetestset "Test the correct setup of the various optimizer caches for MultiHeadAttention" begin
    include("multi_head_attention_stiefel_optim_cache.jl")
end
@safetestset "Volume-Preserving Transformer (skew-symmetric tests)" begin
    include("test_skew_map.jl")
end
@safetestset "Volume-Preserving Transformer (cayley-transform tests)" begin
    include("test_cayley_transforms.jl")
end
@safetestset "Linear Symplectic Attention" begin
    include("linear_symplectic_attention.jl")
end
