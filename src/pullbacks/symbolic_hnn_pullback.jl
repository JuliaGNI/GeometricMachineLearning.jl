"""
    SymbolicPullback(arch::HamiltonianArchitecture)

Make a `SymbolicPullback` of [`HNNLoss`](@ref) for a [`HamiltonianArchitecture`](@ref).

Its value is `HNNLoss` on the batch, ``\\ell = \\|R\\|/\\|Y\\|`` with ``R = X_H(X) - Y``. Its
gradient is ``\\Delta \\sum_i \\nabla_\\theta \\tfrac12\\|r_i\\|^2 / (\\|R\\|\\,\\|Y\\|)``: the generated
per-sample gradient of the surrogate ``\\tfrac12\\|r_i\\|^2``, summed over the batch and scaled by the
two norms of the value's residual. The gradient has the parameters' element type, and a zero
residual gives a zero gradient.

`SymbolicNeuralNetworks.SymbolicPullback(nn, loss)` cannot be used: it sizes the target of the loss
with `output_dimension(nn.model)`, the scalar Hamiltonian, but `HNNLoss` compares against the vector
field, which has the dimension of the input.
"""
function SymbolicPullback(arch::HamiltonianArchitecture)
    nn = SymbolicNeuralNetwork(arch)
    hvf = symbolic_hamiltonian_vector_field(nn)
    # `inplace = false`, as in `hamiltonian_vector_field`: `HNNLoss` is also differentiated by `Zygote`.
    loss = HNNLoss(SymbolicNeuralNetworks.build_nn_function(
        hvf, nn.params, nn.input; inplace = false))
    soutput = Symbolics.variables(:y, 1:input_dimension(nn.model))
    gradient = SymbolicNeuralNetworks.symbolic_parameter_gradient(
        sum(abs2, hvf - soutput) / 2, nn)
    gradient_function = SymbolicNeuralNetworks.build_nn_function(
        gradient, nn.params, nn.input, soutput; reduce = +)
    SymbolicPullback(loss, _HNNParameterGradient(gradient_function))
end

# The `fun` of the HNN pullback; it holds the generated gradient and selects the call method below.
struct _HNNParameterGradient{FT}
    gradient_function::FT
end

# More specific than `SymbolicNeuralNetworks`' call in every argument, which computes the value and
# the gradient separately and would evaluate `hvf` twice.
function (pb::SymbolicPullback{<:HNNLoss, <:_HNNParameterGradient})(
        ps::NetworkParameters, model, (input, output)::Tuple{<:QPTOAT, <:QPTOAT})
    norm_R = norm(pb.loss.hvf(input, ps) - output)
    norm_Y = norm(output)
    # `pinv`: a zero residual gives the zero gradient, not `NaN`. Each norm divides each entry on its
    # own, because `‖R‖‖Y‖` overflows, and its inverse underflows, where neither norm does.
    inv_R = pinv(norm_R)
    TP = parameter_eltype(ps)
    gradient_function = pb.fun.gradient_function
    norm_R / norm_Y,
    Δ -> mapstorage(x -> TP.(x .* Δ .* inv_R ./ norm_Y), gradient_function(input, output, ps))
end
