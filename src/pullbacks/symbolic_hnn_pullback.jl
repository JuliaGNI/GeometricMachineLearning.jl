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
    _check_symbolic_pullback_is_tractable(arch)
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

# How many `SymplecticEuler` layers the architecture stacks. Only the generalized architectures
# have more than one; everything else traces as a single pass.
_n_symplectic_integrators(::Any) = 1
_n_symplectic_integrators(arch::GeneralizedHamiltonianArchitecture) = arch.n_integrators

@doc raw"""
    _check_symbolic_pullback_is_tractable(arch)

Throw if building a `SymbolicPullback` for `arch` would not finish.

`SymbolicPullback` traces the whole chain symbolically, and each `SymplecticEuler` layer inlines the
symbolic gradient of its energy network — an expression that is itself already a derivative. Stacking
integrators inlines that expression inside itself, so it grows *multiplicatively* rather than
additively. Measured at `dim = 4, width = 4, nhidden = 1`:

| `n_integrators` | symbolic loss | its parameter derivative | build time |
|---|---|---|---|
| 1 | 3.4 ⋅ 10⁵ characters | 1.5 ⋅ 10⁸ characters | ≈ 1.4 s |
| 2 | 1.4 ⋅ 10⁹ characters | — | does not finish, past 8 GB |

So one integrator is fine — and worth it, the built function evaluates about 100 times faster than
the `Zygote` pullback — while two are hopeless. Rather than let that look like a hang, refuse it.

The way out is not to trace the inlined chain at all, but to compose the pullback layer by layer.
`SymbolicNeuralNetworks` 0.6 added that construction
([SNN #49](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/49)) and 0.7 widened its seam
so that it reaches a `SymplecticEuler`, which threads the parameters of the system on to the next layer
and therefore returns a `Tuple` rather than an array
([SNN #54](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl/issues/54); the layer says how it
meets the seam in `src/architectures/generalized_hamiltonian_neural_network.jl`). So

```julia
SymbolicNeuralNetworks.SymbolicPullback(SymbolicNeuralNetwork(arch), FeedForwardLoss())
```

builds for any `n_integrators` — four steps for two integrators in about 1.5 s, against an expression
of 10⁹ terms that never finished — and for an architecture built with `parameters` it is the *only*
construction that builds, since the monolithic one traces the chain from a plain vector and the layers
then default the system parameters away.

This function guards the two constructors *in this file*, which still build one expression for the
whole network by hand, and the limit is still real for them:

- `SymbolicPullback(arch)` uses [`HNNLoss`](@ref), which never applies the model — it evaluates the
  pre-built Hamiltonian vector field and reads the parameters directly. It is therefore not a function
  of the chain's prediction at all, which is what a layerwise seed has to start from, so the layerwise
  construction cannot express it.
- `SymbolicPullback(nn, loss, system_params)` predates the widened seam. [`ParametricLoss`](@ref) now
  declares its `loss_expression`, so the upstream constructor can seed it; migrating this one onto it
  would also retire the input concatenation below, and is
  [issue #245](https://github.com/JuliaGNI/GeometricMachineLearning.jl/issues/245)'s remaining half.
"""
function _check_symbolic_pullback_is_tractable(arch)
    n = _n_symplectic_integrators(arch)
    n > 1 && throw(ArgumentError(
        "cannot build *this* `SymbolicPullback` for an architecture with $(n) integrators: it builds " *
        "one symbolic expression for the whole network, which grows multiplicatively with " *
        "`n_integrators` and already exceeds 10⁹ terms at two, so the build does not finish. " *
        "`SymbolicNeuralNetworks.SymbolicPullback(SymbolicNeuralNetwork(arch), FeedForwardLoss())` " *
        "composes the pullback layer by layer instead and has no such limit; `ZygotePullback(loss)`, " *
        "which is what `Optimizer` uses by default, is the other option. See " *
        "GeometricMachineLearning issue #245 and SymbolicNeuralNetworks issues #49 and #54."))
    nothing
end

@doc raw"""
    SymbolicPullback(nn, loss, system_params)

The `SymbolicPullback` for a network whose forward pass also takes the parameters of the *system*,
i.e. one built on [`GeneralizedHamiltonianArchitecture`](@ref), with a [`ParametricLoss`](@ref).

# Implementation

This is `SymbolicNeuralNetworks.SymbolicPullback(nn, loss)` with the system parameters threaded
through. `build_nn_function` generates a function of *one* input array, so the flattened system
parameters are appended to the network input, and the symbolic expression splits them off again with
[`_flatten_system_parameters`](@ref) and `unflatten`. The numeric side does the same concatenation,
in the call operators below.

`reduce = +`: the loss of a batch is the sum of the losses of its samples, so its gradient is the
sum of the per-sample gradients.
"""
function SymbolicPullback(nn::NeuralNetwork, loss::ParametricLoss,
        system_params::OptionalParameters; cse::Bool = true, inplace::Bool = true)
    _check_symbolic_pullback_is_tractable(nn.architecture)
    symbolic_system_parameters = SymbolicNeuralNetworks.symbolic_variables(system_params, :S)
    symbolic_network_parameters = SymbolicNeuralNetworks.symbolic_variables(params(nn), :W)

    input_dim = input_dimension(nn.model)
    _, parameter_layout = _flatten_system_parameters(SymbolicNeuralNetworks.Symbolics.Num,
                                                     symbolic_system_parameters)
    sinput = Symbolics.variables(:x, 1:(input_dim + length(system_params)))
    soutput = Symbolics.variables(:y, 1:output_dimension(nn.model))
    symbolic_system_input = unflatten(parameter_layout, sinput[(input_dim + 1):end])

    symbolic_loss = loss(nn.model, symbolic_network_parameters, sinput[1:input_dim], soutput,
                         symbolic_system_input)
    differentials = SymbolicNeuralNetworks.symbolic_differentials(symbolic_network_parameters)
    gradient = SymbolicNeuralNetworks.symbolic_derivative(symbolic_loss, differentials)
    gradient_function = SymbolicNeuralNetworks.build_nn_function(
        gradient, symbolic_network_parameters, sinput, soutput;
        reduce = +, cse = cse, inplace = inplace)
    SymbolicPullback(loss, SymbolicNeuralNetworks.ParameterGradient(gradient_function))
end

# TODO: type piracy -- `SymbolicPullback` is `SymbolicNeuralNetworks`', and so is every argument
# type here. These belong upstream, together with a `build_nn_function` that takes more than one
# data argument, which is what would make the concatenation below unnecessary.
#
# The generated pullback takes *one* input array, so the system parameters are appended to the
# network input before it is called; the loss, which knows about them, gets them separately.
function (_pullback::SymbolicPullback)(ps, model,
        input_output_params::Tuple{<:AbstractMatrix, <:AbstractMatrix,
                                   <:Union{NamedTuple, AbstractVector}})::Tuple
    input, output, system_params = input_output_params
    _pullback.loss(model, ps, input, output, system_params),
        _pullback.fun(concatenate_array_with_parameters(input, system_params), output, ps)
end

# A batch with a time axis: the network is applied sample-wise, so the time and parameter axes are
# folded into one before the pullback sees them.
function (_pullback::SymbolicPullback)(ps, model,
        input_output_params::Tuple{AT, AT, <:Union{NamedTuple, AbstractVector}})::Tuple where {T, AT <: AbstractArray{T, 3}}
    input, output, system_params = input_output_params
    _input = reshape(input, size(input, 1), size(input, 2) * size(input, 3))
    _output = reshape(output, size(output, 1), size(output, 2) * size(output, 3))
    _pullback(ps, model, (_input, _output, system_params))
end

function (_pullback::SymbolicPullback)(ps, model,
        input_output_params::Tuple{<:QPT, <:QPT, <:Union{NamedTuple, AbstractVector}})::Tuple
    input, output, system_params = input_output_params
    _pullback(ps, model, (vcat(input.q, input.p), vcat(output.q, output.p), system_params))
end
