using GeometricMachineLearning, CUDA, Test, KernelAbstractions
using GeometricMachineLearning: ResNet
model = ResNet(4, tanh)

ps = NeuralNetwork(CUDABackend(), Float32, model).params
@test typeof(ps.weight) <: CuArray
@test typeof(ps.bias) <: CuArray
