using Lux, Zygote, CUDA, Random, GeometricMachineLearning, Test, GPUArrays

model = Chain(Dense(10, 20, tanh), Dense(20,20, tanh), Dense(20,5, use_bias=false))

function test_differential(dev::Device, T)
    ps, st = Lux.setup(dev, Random.default_rng(), model)
    @time dg = Zygote.gradient(ps -> sum(Lux.apply(model, convert_to_dev(dev, rand(T, 10)), ps, st)[1]), ps)[1]

    if dev == CUDA.device()
        for layer_name in keys(dg)
            for param in keys(dg[layer_name])
                @test typeof(dg[layer_name][param]) <: AbstractGPUArray
            end
        end
    end
end

T = Float32
for i in 1:5
    test_differential(CPUDevice(), T)
    test_differential(CUDA.device(), T)
end