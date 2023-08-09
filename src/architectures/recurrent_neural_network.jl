
struct RecurrentNeuralNetwork{TAO, TAM} <: Architecture
    dimin::Int
    dimout::Int
    width::Int
    nhidden::Int
    act_output::TAO
    act_rec::TAM

    function RecurrentNeuralNetwork(dimin, dimout; width = dimin, nhidden = 1, activation_output = tanh, activation_recurent = activation_input)
        new{typeof(activation_output), typeof(activation_recurent)}(dimin, dimout, width, nhidden, activation_output, activation_recurent)
    end
end

@inline AbstractNeuralNetworks.dim(arch::RecurrentNeuralNetwork) = arch.dimin

function Chain(nn::RecurrentNeuralNetwork)
    inner_layers = Tuple(
        [RNNLayer(nn.width, nn.width, act_output, act_rec) for _ in 1:nn.nhidden]
    )
    Chain(
        RNNLayer(nn.width, nn.width, act_output, act_rec),
        inner_layers...,
        Linear(nn.width, nn.dimout; use_bias = false)
    )
end

