using Lux, Zygote, CUDA, Random, GeometricMachineLearning, Test, GPUArrays

model = Chain(Dense(10, 20, tanh), Dense(20,20, tanh), Dense(20,5, use_bias=false))
ps_gpu, st_gpu = Lux.setup(CUDa.device(), Random.default_rng(), model)

loss(x, ps, st) = sum(Lux.apply(model, x, ps, st)[1])

function kernel(x)
    i = threadIdx().x
    x[i] = loss(CUDA.rand(10), ps, st)
    return 
end

x = CUDA.zeros(100)

@cuda threads=length(x) kernel(x) 
