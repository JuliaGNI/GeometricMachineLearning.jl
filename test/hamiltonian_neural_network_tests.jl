using GeometricMachineLearning
using Test
using Zygote
import Random 
Random.seed!(1234)

const GMLA = GeometricMachineLearning.Activation
function allocate_network_and_data_loader(dim::Integer, width::Integer, nhidden::Integer, activation::GMLA)
    arch = StandardHamiltonianArchitecture(dim, width, nhidden, activation)
    loss = HNNLoss(arch)
    input = rand(dim, 10)
    output = rand(dim, 10)
    dl = DataLoader(input, output)
    NeuralNetwork(arch), loss, dl
end

"""
This tests if we can call the HNN loss.
"""
function test_hnn_loss( dim::Integer = 2, 
                        width::Integer = dim, 
                        nhidden::Integer = 1, 
                        activation::GMLA = GeometricMachineLearning.SigmoidActivation())
    nn, loss, dl = allocate_network_and_data_loader(dim, width, nhidden, activation)
    @test typeof(loss(nn.params, dl.input, dl.output)) <: Real
    @test typeof(loss) <: NetworkLoss
end

test_hnn_loss()

"""
This tests if we can differentiate the HNN loss.
"""
function test_hnn_loss_derivative(  dim::Integer = 2,
                                    width::Integer = dim,
                                    nhidden::Integer = 1,
                                    activation::GMLA = GeometricMachineLearning.SigmoidActivation())
    nn, loss, dl = allocate_network_and_data_loader(dim, width, nhidden, activation)
    @test typeof(Zygote.gradient(ps -> loss(ps, dl.input, dl.output), nn.params)[1].params) <: NamedTuple
end