using SafeTestsets

@safetestset "Optimizer #2" begin
    include("optimization_step.jl")
end
@safetestset "Optimizer #3" begin
    include("svd_optim.jl")
end
@safetestset "Optimizer #4" begin
    include("psd_optim.jl")
end
@safetestset "Check if Adam with decay converges" begin
    include("adam_with_learning_rate_decay.jl")
end
@safetestset "Gradient optimizer tests" begin
    include("gradient_optimizer.jl")
end
@safetestset "Momentum optimizer tests" begin
    include("momentum_optimizer.jl")
end
@safetestset "Optimizers with structured (non-manifold) weights" begin
    include("structured_array_parameters.jl")
end
@safetestset "_GMLGradient dispatch" begin
    include("gml_gradient_dispatch.jl")
end
@safetestset "The GO-native leaf step scales in the parameter's own element type" begin
    include("step_size_element_type.jl")
end
