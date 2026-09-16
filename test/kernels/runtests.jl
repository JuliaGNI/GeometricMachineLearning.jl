using SafeTestsets

@safetestset "Custom tensor matrix multiplication" begin
    include("tensor_mat_mul.jl")
end
@safetestset "Custom AD rules for kernels" begin
    include("kernel_pullbacks.jl")
end
@safetestset "Test parallel inverses" begin
    include("tensor_inverse.jl")
end
@safetestset "Test parallel Cayley" begin
    include("tensor_cayley.jl")
end
