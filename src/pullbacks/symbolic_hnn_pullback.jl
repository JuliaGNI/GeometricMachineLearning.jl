"""
    SymbolicPullback(arch::HamiltonianArchitecture)

Make a `SymbolicPullback` based on a [`HamiltonianArchitecture`](@ref).

# Implementation

Internally this is calling `SymbolicNeuralNetwork`, [`HNNLoss`](@ref) and
`SymbolicNeuralNetworks.build_nn_function`.

The value of the pullback is [`HNNLoss`](@ref) on the batch,

```math
\\ell = \\|R\\| \\,/\\, \\|Y\\|, \\qquad R = X_H(X) - Y,
```

with Frobenius norms over the whole batch. Its gradient is

```math
\\Delta \\cdot \\frac{\\sum_i \\nabla_\\theta \\tfrac12\\|r_i\\|^2}{\\|R\\|\\,\\|Y\\|},
```

where ``r_i`` is the residual of sample ``i`` and ``\\nabla_\\theta\\tfrac12\\|r_i\\|^2`` its
per-sample gradient. The generated function of the sum of those surrogates is the gradient of the
*per-sample sum*, which is not the gradient of ``\\ell``: ``\\ell`` divides by the norm of the
whole batch's residual, so every sample's gradient carries one scalar in common. The pullback takes
that scalar from the residual of its value, so it evaluates the Hamiltonian vector field once per
call. The scale divides by each norm and never forms the product ``\\|R\\|\\|Y\\|``, which overflows
where neither norm does.

`SymbolicNeuralNetworks.SymbolicPullback(nn, loss)` cannot be used here. It sizes the symbolic
target of the loss with `output_dimension(nn.model)`, which for a Hamiltonian network is `1` — the
scalar Hamiltonian. [`HNNLoss`](@ref) does not compare against that: it compares against the
Hamiltonian *vector field*, which has the dimension of the network's *input*. The construction
below builds one generated gradient of the surrogate with
`SymbolicNeuralNetworks.build_nn_function`, and the pullback's call scales it by the batch's two
norms. `ps` is a `NetworkParameters`, as for a [`ZygotePullback`](@ref), and the gradient has the
parameters' element type whatever the data's.

A batch whose residual is zero — data the network's own vector field generates — gives a zero
gradient, as [`ZygotePullback`](@ref)'s does: the scale is `pinv(\\|R\\|)`, which is zero at zero.
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

"""
    _HNNParameterGradient(gradient_function)

The `fun` of a [`SymbolicPullback`](@ref) of [`HNNLoss`](@ref): the generated per-sample gradient of
the surrogate, which the pullback's call scales by the batch's two norms.
"""
struct _HNNParameterGradient{FT}
    gradient_function::FT
end

# More specific than `SymbolicNeuralNetworks`' call in every argument, which computes the value and
# the gradient separately and would evaluate `hvf` twice.
function (pb::SymbolicPullback{<:HNNLoss, <:_HNNParameterGradient})(
        ps::NetworkParameters, model, (input, output)::Tuple{<:QPTOAT, <:QPTOAT})
    norm_R = norm(pb.loss.hvf(input, ps) - output)
    norm_Y = norm(output)
    # `pinv` and not `1 / norm_R`: at a zero residual the per-sample gradients are zero themselves,
    # and `pinv(0)` is zero, so their product is the zero set rather than `NaN`. That is what
    # ChainRules' `norm` pullback gives `Zygote` at the same point. Each norm divides every gradient
    # entry, and the seed is not folded into the one scalar `1/(‖R‖‖Y‖)`: in `Float32` that scalar is
    # subnormal once `‖R‖‖Y‖` passes `1/floatmin(Float32) ≈ 8.5e37`, where the per-entry divisions
    # keep every leaf finite and non-zero. The generated gradient has the data's element type, so
    # each leaf is converted to the parameters'.
    inv_R = pinv(norm_R)
    TP = parameter_eltype(ps)
    gradient_function = pb.fun.gradient_function
    norm_R / norm_Y,
    Δ -> mapstorage(x -> TP.(x .* Δ .* inv_R ./ norm_Y), gradient_function(input, output, ps))
end
