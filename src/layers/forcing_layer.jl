const f_nhidden_default = 1
const f_activation_default = tanh

# What the forcing of a layer depends on. All three types add to `p`, the component a forcing or a
# dissipation enters; the `Q` and `P` suffixes name the component the forcing *reads*.
const FORCING_TYPES = (:Q, :P, :QP)

function _check_forcing_type(type::Symbol)
    type in FORCING_TYPES || throw(ArgumentError(
        "the forcing type is `:$(type)`; it is `:Q`, `:P` or `:QP`."))
    type
end

# A forcing layer splits the state into halves of `dim ÷ 2` rows, which an odd `dim` has no split
# for; the refusal is here rather than at the first call.
function _check_forcing_dimension(dim::Integer)
    iseven(dim) || throw(ArgumentError(
        "a forcing layer takes the full phase-space dimension, which is even; it is $(dim)."))
    dim
end

@doc raw"""
    ForcingLayer <: AbstractExplicitLayer

A layer that adds a non-conservative term — a forcing or a dissipation — to the ``p`` component of
the state, and leaves the ``q`` component unchanged.

A forcing enters the ``\dot{p}`` equation, so every forcing layer adds to ``p``. The three types of
forcing layer differ in what the forcing *reads*: a forcing layer composes a `Chain` of its
own with the identity on ``q``,

```math
\begin{pmatrix} q \\ p \end{pmatrix} \mapsto
\begin{pmatrix} q \\ p + f([z; \mu]) \end{pmatrix},
```

where ``f`` is the layer's network, ``\mu`` the flattened parameters of the system, and ``z`` is
``q`` for a [`ForcingLayerQ`](@ref), ``p`` for a [`ForcingLayerP`](@ref), and ``[q; p]`` for a
[`ForcingLayerQP`](@ref). The network's last layer has `dim ÷ 2` rows, so the term has the length
of `p` whatever the type.

The step this adds a term to is the symplectic-Euler step of a [`GradientLayerP`](@ref) of a
[`GSympNet`](@ref); [`ForcedSympNet`](@ref) is the architecture that puts one in front of every
such step.

# Constructor

`ForcingLayer(dim, width, nhidden, activation; parameters, return_parameters, type)` takes the
arguments and the defaults of [`ForcingLayerQ`](@ref), and `type` is `:Q`, `:P` or `:QP`, `:P` by
default. `dim` is the full phase-space dimension, as in [`GradientLayerQ`](@ref)`(dim, …)`: the
forcing has `dim ÷ 2` rows. Use the constructors [`ForcingLayerQ`](@ref), [`ForcingLayerP`](@ref)
and [`ForcingLayerQP`](@ref), which fix the type, unless you build one from a
[`ForcedSympNet`](@ref).
"""
struct ForcingLayer{M, N, type, ReturnParameters, MT <: Chain} <:
       AbstractExplicitLayer{M, N}
    dim::Int
    width::Int
    nhidden::Int
    model::MT
end

# The network of a forcing layer: a first dense layer over the state the forcing reads and the
# flattened system parameters, `nhidden` square hidden layers, and a linear map back to the
# `dim ÷ 2` rows of the forcing.
function _forcing_model(dim::Integer, width::Integer, nhidden::Integer,
        parameter_length::Integer, activation, type::Symbol)
    input_dim = type === :QP ? dim + parameter_length : dim ÷ 2 + parameter_length
    Chain(Dense(input_dim, width, activation),
        ntuple(_ -> Dense(width, width, activation), nhidden)...,
        Linear(width, dim ÷ 2))
end

function ForcingLayer(dim::Integer, width::Integer = dim,
        nhidden::Integer = f_nhidden_default, activation = f_activation_default;
        parameters::OptionalParameters = NullParameters(),
        return_parameters::Bool = false, type::Symbol = :P)
    _check_forcing_dimension(dim)
    _check_forcing_type(type)
    # The parameters of the system fix the length of the flat vector the layer takes beside its
    # state. They are not stored: the `μ` the layer is called with is the one that enters.
    parameter_length = length(first(_flatten_system_parameters(parameters)))
    model = _forcing_model(
        Int(dim), Int(width), Int(nhidden), parameter_length, activation, type)
    ForcingLayer{Int(dim), Int(dim), type, return_parameters, typeof(model)}(
        Int(dim), Int(width), Int(nhidden), model)
end

"""
    ForcingLayerQ(dim, width = dim, nhidden = $(f_nhidden_default), activation = $(f_activation_default); parameters, return_parameters)

A forcing layer whose forcing depends only on the `q` component. It changes `p` and leaves `q`
alone; see [`ForcingLayer`](@ref).

# Arguments

- `dim::Integer`: the full phase-space dimension, which is even. The forcing has `dim ÷ 2` rows.
- `width::Integer = dim`: the width of the layer's network.
- `nhidden::Integer = $(f_nhidden_default)`: the number of hidden layers of the network.
- `activation = $(f_activation_default)`: the activation function of the network.

# Keyword arguments

- `parameters = NullParameters()`: the parameters of the system, which set the length of the flat
  vector the layer takes beside the state it reads.
- `return_parameters::Bool = false`: whether the layer returns `(output, μ)` instead of the output,
  so that the layer after it in a `Chain` receives `μ` too.

# Examples

```julia
ForcingLayerQ(4, 8, 1, tanh; parameters = (m = 1.0,), return_parameters = false)
```
"""
const ForcingLayerQ{M, N, ReturnParameters} = ForcingLayer{M, N, :Q, ReturnParameters}

"""
    ForcingLayerP(dim, width = dim, nhidden = $(f_nhidden_default), activation = $(f_activation_default); parameters, return_parameters)

A forcing layer whose forcing depends only on the `p` component. It changes `p` and leaves `q`
alone; see [`ForcingLayerQ`](@ref) for the arguments.
"""
const ForcingLayerP{M, N, ReturnParameters} = ForcingLayer{M, N, :P, ReturnParameters}

"""
    ForcingLayerQP(dim, width = dim, nhidden = $(f_nhidden_default), activation = $(f_activation_default); parameters, return_parameters)

A forcing layer whose forcing depends on the `q` and the `p` component. It changes `p` and leaves
`q` alone; see [`ForcingLayerQ`](@ref) for the arguments.
"""
const ForcingLayerQP{M, N, ReturnParameters} = ForcingLayer{M, N, :QP, ReturnParameters}

function ForcingLayerQ(dim::Integer, width::Integer = dim,
        nhidden::Integer = f_nhidden_default, activation = f_activation_default;
        parameters::OptionalParameters = NullParameters(),
        return_parameters::Bool = false)
    ForcingLayer(dim, width, nhidden, activation; parameters, return_parameters, type = :Q)
end

function ForcingLayerP(dim::Integer, width::Integer = dim,
        nhidden::Integer = f_nhidden_default, activation = f_activation_default;
        parameters::OptionalParameters = NullParameters(),
        return_parameters::Bool = false)
    ForcingLayer(dim, width, nhidden, activation; parameters, return_parameters, type = :P)
end

function ForcingLayerQP(dim::Integer, width::Integer = dim,
        nhidden::Integer = f_nhidden_default, activation = f_activation_default;
        parameters::OptionalParameters = NullParameters(),
        return_parameters::Bool = false)
    ForcingLayer(dim, width, nhidden, activation; parameters, return_parameters, type = :QP)
end

parameterlength(l::ForcingLayer) = parameterlength(l.model)

# The layer's parameters are the parameters of its own `Chain`, which is a `NetworkParameters` with
# a level of its own. A `Chain` applies a layer with whatever its `initialparameters` returned, and
# the parameter tree walks a nested set, so the nesting needs nothing else.
function initialparameters(rng::Random.AbstractRNG,
        init_weight::AbstractNeuralNetworks.Initializer, l::ForcingLayer,
        backend::Backend, ::Type{T}) where {T}
    initialparameters(rng, init_weight, l.model, backend, T)
end

# The state a forcing layer reads: the forcing of a `:Q` layer depends on `q`, of a `:P` layer on
# `p`, and of a `:QP` layer on both, `q` above `p`.
_forcing_input(::Val{:Q}, z) = z.q
_forcing_input(::Val{:P}, z) = z.p
_forcing_input(::Val{:QP}, z) = vcat(z.q, z.p)

# The forward pass on a `(q, p)` state with the system parameters beside it. `ps` is untyped and
# forwarded to the layer's `Chain`: the chain says what it takes, and for this layer that is the
# `NetworkParameters` its own `initialparameters` returned.
function _forcing_forward(l::ForcingLayer{M, M, FT, ReturnParameters}, z, μ,
        ps) where {M, FT, ReturnParameters}
    forcing = l.model(concatenate_array_with_parameters(_forcing_input(Val(FT), z), μ), ps)
    output = (q = z.q, p = z.p + forcing)
    ReturnParameters ? (output, μ) : output
end

function (l::ForcingLayer)(z::NamedTupleOfArrays{<:Any, (:q, :p)}, ps)
    _forcing_forward(l, z, NullParameters(), ps)
end

function (l::ForcingLayer)(
        z_and_μ::Tuple{<:NamedTupleOfArrays{<:Any, (:q, :p)}, <:OptionalParameters}, ps)
    _forcing_forward(l, z_and_μ..., ps)
end

# A stacked state has its `(q, p)` split taken here, and the halves stacked again on the way out.
function _split_state(l::ForcingLayer{M, M}, x::AbstractArray) where {M}
    size(x, 1) == M || throw(DimensionMismatch(
        "a forcing layer of dimension $M takes a state of $M rows; this one has $(size(x, 1))."))
    assign_q_and_p(x, M ÷ 2)
end

_stacked_output(output::NamedTuple) = vcat(output.q, output.p)
function _stacked_output(output_and_μ::Tuple)
    output = first(output_and_μ)
    (vcat(output.q, output.p), last(output_and_μ))
end

function (l::ForcingLayer{M, M})(x::AbstractArray, ps) where {M}
    _stacked_output(_forcing_forward(l, _split_state(l, x), NullParameters(), ps))
end

function (l::ForcingLayer{M, M})(
        x_and_μ::Tuple{<:AbstractArray, <:OptionalParameters}, ps) where {M}
    x, μ = x_and_μ
    _stacked_output(_forcing_forward(l, _split_state(l, x), μ, ps))
end
