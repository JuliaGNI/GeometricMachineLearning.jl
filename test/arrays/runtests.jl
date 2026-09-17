using SafeTestsets

@safetestset "Symplectic Potential (array tests)" begin
    include("poisson_tensor.jl")
end
@safetestset "Test triangular matrices" begin
    include("triangular.jl")
end
