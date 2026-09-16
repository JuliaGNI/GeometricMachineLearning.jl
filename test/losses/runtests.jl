using SafeTestsets

@safetestset "Symplectic Euler and variational midpoint losses" begin
    include("training_method_losses.jl")
end
@safetestset "Test NetworkLoss + Optimizer" begin
    include("losses_and_optimization.jl")
end
