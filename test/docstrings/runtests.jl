using SafeTestsets

@safetestset "Data loader docstring examples" begin
    include("data_loader.jl")
end
@safetestset "Layer and architecture docstring examples" begin
    include("layers_and_architectures.jl")
end
@safetestset "Loss docstring examples" begin
    include("losses.jl")
end
@safetestset "Manifold docstring examples" begin
    include("manifolds.jl")
end
@safetestset "Utility and pullback docstring examples" begin
    include("utilities.jl")
end
