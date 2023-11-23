using GeometricMachineLearning
using Test
using KernelAbstractions
import Random, Zygote

Random.seed!(1234)

function test_gradient_layer_application(T, M, N, batch_size=10)
    dummy_model = GradientLayerQ(M, N, tanh)
    ps = initialparameters(CPU(), T, dummy_model)

    x = rand(T, M)
    x_applied = dummy_model(x, ps)
    @test typeof(x_applied) == typeof(x)
    @test size(x_applied) == size(x)

    X = rand(T, M, batch_size)
    X_applied = dummy_model(X, ps)
    @test typeof(X_applied) == typeof(X)
    @test size(X_applied) == size(X)
end

function test_gradient_layer_derivative_and_update(T, M, N, batch_size=10)
    dummy_model = Chain(GradientLayerP(M, N, tanh), GradientLayerQ(M, N, tanh))
    ps = initialparameters(CPU(), T, dummy_model)
    o = Optimizer(AdamOptimizer(T(0.1), T(.9), T(0.999), T(3e-7)), ps)

    # test for vector 
    x = rand(T, M)
    gs = Zygote.gradient(ps -> sum(dummy_model(x, ps)), ps)[1]
    optimization_step!(o, dummy_model, ps, gs)
    
    # test for matrix 
    X = rand(T, M, batch_size)
    gs = Zygote.gradient(ps -> sum(dummy_model(X, ps)), ps)[1]
    optimization_step!(o, dummy_model, ps, gs)
end


types = (Float32, Float64)
for T in types
    for M in 4:2:10
        for N in M:2:2*M 
            test_gradient_layer_application(T, M, N)
            test_gradient_layer_derivative_and_update(T, M, N)
        end
    end
end