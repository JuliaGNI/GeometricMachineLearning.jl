using GeometricMachineLearning, Test, Zygote

function test_data_loader(sys_dim, n_time_steps, n_params, T=Float32)
    data = randn(T, sys_dim, n_time_steps, n_params)
    dl = DataLoader(data)

    # first argument is sys_dim, second is number of heads, third is number of units
    model = Transformer(dl.input_dim, 2, 1)
    ps = initialparameters(CPU(), T, model)
    dx = Zygote.gradient(ps -> GeometricMachineLearning.loss(model, ps, dl), ps)[1]
    ps_copy = deepcopy(ps)
    o = Optimizer(GradientOptimizer(), ps)
    optimization_step!(o, model, ps, dx)
    @test ps !== ps_copy    
end

test_data_loader(4, 200, 1000)