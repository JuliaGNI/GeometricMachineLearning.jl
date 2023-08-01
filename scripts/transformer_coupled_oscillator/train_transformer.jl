using GeometricMachineLearning, KernelAbstractions

include("generate_data.jl")

data = generate_data()
backend = CPU()

attention_window = 8
model = Chain   (MultiHeadAttention(4,2,Stiefel=true),
                ResNet(4,tanh),
                MultiHeadAttention(4,2,Stiefel=true),
                ResNet(4))
ps = initialparameters(backend())