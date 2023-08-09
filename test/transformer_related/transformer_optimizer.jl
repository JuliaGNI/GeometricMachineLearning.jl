using Test, KernelAbstractions, GeometricMachineLearning, Zygote, LinearAlgebra

function transformer_gradient_test(T, dim, n_heads, L, seq_length=8, batch_size=10)
    model = Chain(Transformer(dim, n_heads, L, Stiefel=true), ResNet(dim))
    model = Transformer(dim, n_heads, L, Stiefel=true)

    ps = initialparameters(KernelAbstractions.CPU(), T, model)
    
    input = rand(T, dim, seq_length, batch_size)
    
    loss(ps, input) = norm(model(input, ps))
    dx = Zygote.gradient(ps -> loss(ps, input), ps)[1]

    o₁ = Optimizer(GradientOptimizer(), ps)
    o₂ = Optimizer(MomentumOptimizer(), ps)
    o₃ = Optimizer(AdamOptimizer(), ps)

    ps₁ = deepcopy(ps)
    ps₂ = deepcopy(ps)
    ps₃ = deepcopy(ps)

    optimization_step!(o₁, model, ps₁, dx)
    optimization_step!(o₂, model, ps₂, dx)
    optimization_step!(o₃, model, ps₃, dx)
    @test typeof(ps₁) == typeof(ps₂) == typeof(ps₃) == typeof(ps)
end

transformer_gradient_test(Float32, 10, 5, 4)