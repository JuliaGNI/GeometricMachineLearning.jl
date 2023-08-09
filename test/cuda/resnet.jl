using GeometricMachineLearning, CUDA, Test, KernelAbstractions
using GeometricMachineLearning: ResNet
model = ResNet(4, tanh)

ps = initialparameters(CUDABackend(), Float32, model)
@test typeof(ps.weight) <: CuArray
@test typeof(ps.bias) <: CuArray
