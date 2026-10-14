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
that scalar from the forward pass, which is why it evaluates the Hamiltonian vector field one time
more than the loss value does. The scale divides by each norm and never forms the product
``\\|R\\|\\|Y\\|``, which overflows where neither norm does.

`SymbolicNeuralNetworks.SymbolicPullback(nn, loss)` cannot be used here. It sizes the symbolic
target of the loss with `output_dimension(nn.model)`, which for a Hamiltonian network is `1` — the
scalar Hamiltonian. [`HNNLoss`](@ref) does not compare against that: it compares against the
Hamiltonian *vector field*, which has the dimension of the network's *input*. The construction
below keeps the upstream shape — one generated gradient of the surrogate, built with
`SymbolicNeuralNetworks.build_nn_function` — and carries the batch's two norms in a callable of
this package.

The pullback is a `SymbolicPullback(loss, fun)`. Its `loss` field is the [`HNNLoss`](@ref) of the
architecture, whose value on the batch the pullback returns; its `fun` field is the callable the
gradient comes from. `fun(input, output, ps)` evaluates the Hamiltonian vector field once, for the
residual's norm, and returns a closure over the output sensitivities, so
`fun(input, output, ps)(Δ)` is the gradient that the seed `Δ` produces. A batched `ps` is a
`NetworkParameters`, as a `ZygotePullback`'s is.

A batch whose residual is zero — data the network's own vector field generates — gives a zero
gradient, as [`ZygotePullback`](@ref)'s does: the scale is `pinv(\\|R\\|)`, which is zero at zero.
"""
function SymbolicPullback(arch::HamiltonianArchitecture)
    nn = SymbolicNeuralNetwork(arch)
    loss = HNNLoss(arch)
    soutput = Symbolics.variables(:y, 1:input_dimension(nn.model))
    surrogate = sum(abs2, symbolic_hamiltonian_vector_field(nn) - soutput) / 2
    gradient = SymbolicNeuralNetworks.symbolic_parameter_gradient(surrogate, nn)
    gradient_function = SymbolicNeuralNetworks.build_nn_function(
        gradient, nn.params, nn.input, soutput; reduce = +)
    SymbolicPullback(loss, _HNNParameterGradient(loss, gradient_function))
end

"""
    _HNNParameterGradient(loss, gradient_function)

The `fun` of a [`SymbolicPullback`](@ref) of [`HNNLoss`](@ref): it evaluates the Hamiltonian vector
field once, for the residual's norm, and hands back the closure over the output sensitivities.
"""
struct _HNNParameterGradient{HLT, FT} <: Function
    loss::HLT
    gradient_function::FT
end

function (g::_HNNParameterGradient)(input, output, ps)
    # The residual's norm is the one scalar the generated per-sample gradients do not carry, so it
    # is taken from a forward pass here: one evaluation of `hvf` more than the loss value needs.
    norm_R = norm(g.loss.hvf(input, ps) - output)
    # `pinv` and not `1 / norm_R`: at a zero residual the per-sample gradients are zero themselves,
    # and `pinv(0)` is zero, so their product is the zero set rather than `NaN`. That is what
    # ChainRules' `norm` pullback gives `Zygote` at the same point. Each norm divides a gradient
    # entry and not the seed: the one scalar `1/(‖R‖‖Y‖)` goes subnormal in `Float32` from a scale
    # of about `1e40`, where the gradient entries themselves stay of order one.
    Δ -> mapstorage(
        x -> (x * convert(parameter_eltype(ps), Δ)) .* pinv(norm_R) ./ norm(output),
        g.gradient_function(input, output, ps))
end
