using Lux, Zygote, CUDA, Random, GeometricMachineLearning, Test, GPUArrays

input_size = 10

model = Chain(Dense(input_size, 1, tanh), Dense(1,20, tanh), Dense(20,5, use_bias=false))
ps_gpu, st_gpu = Lux.setup(CUDA.device(), Random.default_rng(), model)

loss(input, ps, st) = sum(Lux.apply(model, input, ps, st)[1])

function kernel(x, ps, st, input)
    i = threadIdx().x
    x[i] = loss(input, ps, st)
    return 
end

x = CUDA.zeros(2)

input = CUDA.rand(input_size)
@cuda threads=length(x) kernel(x, ps_gpu, st_gpu, input) 
