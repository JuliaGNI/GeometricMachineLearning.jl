using GeometricMachineLearning, Test, Zygote
import Random 

Random.seed!(1234)

@doc raw"""
This tests the gradient optimizer called together with the `DataLoader` (applied to a tensor).
"""
function test_data_loader(sys_dim, n_time_steps, n_params, T=Float32)
    data = randn(T, sys_dim, n_time_steps, n_params)
    dl = DataLoader(data)

    # first argument is sys_dim, second is number of heads, third is number of units
    model = Transformer(dl.input_dim, 2, 1)
    ps = initialparameters(model, CPU(), T)
    loss = GeometricMachineLearning.TransformerLoss(n_time_steps)
    dx = Zygote.gradient(ps -> loss(model, ps, dl.input, dl.input), ps)[1]
    ps_copy = deepcopy(ps)
    o = Optimizer(GradientOptimizer(), ps)
    λY = GlobalSection(ps)
    optimization_step!(o, λY, ps, dx)
    @test ps !== ps_copy    
end

test_data_loader(4, 200, 1000)