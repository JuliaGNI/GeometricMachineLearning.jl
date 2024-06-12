using Test, KernelAbstractions, GeometricMachineLearning, Zygote, LinearAlgebra
using GeometricMachineLearning: ResNetLayer
import Random 

Random.seed!(1234)

@doc raw"""
This function tests if the `GradientOptimzier`, `MomentumOptimizer`, `AdamOptimizer` and `BFGSOptimizer` act on the neural network weights via `optimization_step!`.
"""
function transformer_gradient_test(T, dim, n_heads, L, seq_length=8, batch_size=10)
    model = Chain(Transformer(dim, n_heads, L, Stiefel=true), ResNetLayer(dim))
    model = Transformer(dim, n_heads, L, Stiefel=true)

    ps = initialparameters(model, KernelAbstractions.CPU(), T)
    
    input = rand(T, dim, seq_length, batch_size)
    
    loss(ps, input) = norm(model(input, ps))
    dx = Zygote.gradient(ps -> loss(ps, input), ps)[1]

    o₁ = Optimizer(GradientOptimizer(), ps)
    o₂ = Optimizer(MomentumOptimizer(), ps)
    o₃ = Optimizer(AdamOptimizer(), ps)
    o₄ = Optimizer(BFGSOptimizer(), ps)

    ps₁ = deepcopy(ps)
    ps₂ = deepcopy(ps)
    ps₃ = deepcopy(ps)
    ps₄ = deepcopy(ps)

    λY₁ = GlobalSection(ps₁)
    λY₂ = GlobalSection(ps₂)
    λY₃ = GlobalSection(ps₃)
    λY₄ = GlobalSection(ps₄)
    optimization_step!(o₁, λY₁, ps₁, dx)
    optimization_step!(o₂, λY₂, ps₂, dx)
    optimization_step!(o₃, λY₃, ps₃, dx)
    optimization_step!(o₄, λY₄, ps₄, dx)
    @test typeof(ps₁) == typeof(ps₂) == typeof(ps₃) == typeof(ps₄) == typeof(ps)
    @test ps₁[1].PQ.head_1 ≉ ps₂[1].PQ.head_1 ≉ ps₃[1].PQ.head_1 ≉ ps₄[1].PQ.head_1 # ≉ ps[1].PQ.head_1
end

transformer_gradient_test(Float32, 10, 5, 4)