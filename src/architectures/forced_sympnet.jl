const fs_n_forcing_layers_default = 2

@doc raw"""
    ForcedSympNet(d)

Make a forced SympNet with dimension ``d.``

A forced SympNet is a [`GSympNet`](@ref) with a [`ForcingLayer`](@ref) in front of every
[`GradientLayerP`](@ref), so that each step which changes ``p`` also adds a learned forcing to it:

```math
\begin{pmatrix} q \\ p \end{pmatrix} \mapsto
\begin{pmatrix} q \\ p + \nabla V(q) + f(z) \end{pmatrix},
```

where ``\nabla V`` is the gradient layer's update and ``f`` the forcing layer's network, which reads
``z`` — ``q``, ``p`` or ``[q; p]``, as `forcing_type` says. A forced SympNet takes no parameters of
the system: the forcing reads the state only. It is a `NeuralNetworkIntegrator` and not a
[`SympNet`](@ref): the forcing breaks symplecticity, which is what lets it model a non-conservative
system.

The layer sequence follows the loop index ``i \in \{1, \ldots, n\_layers\}``: for `init_upper` the
odd ``i`` is a [`GradientLayerQ`](@ref), and the even ``i`` a [`ForcingLayer`](@ref) followed by a
[`GradientLayerP`](@ref); without it, the other way around. The chain therefore holds `n_layers`
gradient layers and one forcing layer per step that changes ``p``.

There exists an additional constructor that can be called by supplying an instance of
[`DataLoader`](@ref).

# Arguments

Keyword arguments are:
- `upscaling_dimension::Int = 2d`: The *upscaling dimension* of the gradient layer. See the
  documentation for [`GradientLayerQ`](@ref) and [`GradientLayerP`](@ref) for further explanation.
  The forcing layers have this width as well.
- `n_layers::Int""" * "$(g_n_layers_default)`" *
     raw""": The number of layers (i.e. the total number of [`GradientLayerQ`](@ref) and
  [`GradientLayerP`](@ref)).
- `n_forcing_layers::Int""" * "$(fs_n_forcing_layers_default)`" *
     raw""": The number of hidden layers of each [`ForcingLayer`](@ref).
- `activation""" * "$(g_activation_default)`" *
     raw""": The activation function that is applied.
- `init_upper::Bool""" * "$(g_init_upper_default)`" *
     raw""": Initialize the gradient layer so that it first modifies the ``q``-component.
- `forcing_type::Symbol = :P`: What the forcing of each [`ForcingLayer`](@ref) reads, `:Q`, `:P` or
  `:QP`.
"""
struct ForcedSympNet{FT, AT} <: NeuralNetworkIntegrator
    dim::Int
    upscaling_dimension::Int
    n_layers::Int
    n_forcing_layers::Int
    activation::AT
    init_upper::Bool

    function ForcedSympNet(dim::Integer; upscaling_dimension = 2 * dim,
            n_layers = g_n_layers_default,
            n_forcing_layers = fs_n_forcing_layers_default,
            activation = g_activation_default,
            init_upper = g_init_upper_default,
            forcing_type::Symbol = :P)
        _check_forcing_dimension(dim)
        _check_forcing_type(forcing_type)
        new{forcing_type, typeof(activation)}(
            Int(dim), Int(upscaling_dimension), Int(n_layers), Int(n_forcing_layers),
            activation, init_upper)
    end
end

ForcedSympNet(dl::DataLoader; kwargs...) = ForcedSympNet(dl.input_dim; kwargs...)

# The alternating gradient layers of `GSympNet`, with a forcing layer in front of every layer that
# changes `p`.
function Chain(arch::ForcedSympNet{FT}) where {FT}
    layers = ()
    for i in 1:arch.n_layers
        layers = if isodd(i) == arch.init_upper
            (layers..., GradientLayerQ(arch.dim, arch.upscaling_dimension, arch.activation))
        else
            (layers...,
                ForcingLayer(arch.dim, arch.upscaling_dimension, arch.n_forcing_layers,
                    arch.activation; type = FT),
                GradientLayerP(arch.dim, arch.upscaling_dimension, arch.activation))
        end
    end
    Chain(layers...)
end
