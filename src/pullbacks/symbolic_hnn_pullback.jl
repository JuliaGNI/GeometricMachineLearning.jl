"""
    SymbolicPullback(arch::HamiltonianArchitecture)

Make a `SymbolicPullback` based on a [`HamiltonianArchitecture`](@ref).

# Implementation

Internally this is calling `SymbolicNeuralNetwork` and [`HNNLoss`](@ref).

`SymbolicNeuralNetworks.SymbolicPullback(nn, loss)` cannot be used here. It sizes the symbolic
target of the loss with `output_dimension(nn.model)`, which for a Hamiltonian network is `1` — the
scalar Hamiltonian. [`HNNLoss`](@ref) does not compare against that: it compares against the
Hamiltonian *vector field*, which has the dimension of the network's *input*. The construction
below is the upstream one with that one dimension corrected.
"""
function SymbolicPullback(arch::HamiltonianArchitecture)
    nn = SymbolicNeuralNetwork(arch)
    loss = HNNLoss(arch)
    soutput = Symbolics.variables(:y, 1:input_dimension(nn.model))
    symbolic_loss = loss(nn.model, nn.params, nn.input, soutput)
    gradient = SymbolicNeuralNetworks.symbolic_parameter_gradient(symbolic_loss, nn)
    # `reduce = +`: the loss of a batch is the sum of the losses of its samples, so its gradient is
    # the sum of the per-sample gradients.
    gradient_function = SymbolicNeuralNetworks.build_nn_function(gradient, nn.params, nn.input,
                                                                 soutput; reduce = +)
    SymbolicPullback(loss, SymbolicNeuralNetworks.ParameterGradient(gradient_function))
end
