@doc raw"""
    ForcedSympNet <: NeuralNetworkIntegrator

`ForcedSympNet`s are based on [`SympNet`](@ref)s [jin2020sympnets](@cite) and include [`ForcingLayer`](@ref)s. They are based on [`GSympNet`](@ref)s.

!!! warn
    For the moment only ``p`` forcing is implemented (i.e. we only use [`ForcingLayerP`](@ref)).

# Constructor

```julia
ForcedSympNet(d)
```

Make a forced SympNet with dimension ``d.``

# Arguments

Keyword arguments are:
- `upscaling_dimension::Int = 2d`: The *upscaling dimension* of the gradient layer. See the documentation for [`GradientLayerQ`](@ref) and [`GradientLayerP`](@ref) for further explanation.
- `n_layers::Int""" * "$(g_n_layers_default)`" * raw""": The number of layers (i.e. the total number of [`GradientLayerQ`](@ref) and [`GradientLayerP`](@ref)).
- `activation""" * "$(g_activation_default)`" * raw""": The activation function that is applied.
- `init_upper::Bool""" * "$(g_init_upper_default)`" * raw""": Initialize the gradient layer so that it first modifies the $q$-component.
"""
struct ForcedSympNet{AT} <: NeuralNetworkIntegrator
    dim::Int
    upscaling_dimension::Int
    n_layers::Int
    act::AT
    init_upper::Bool

    function ForcedSympNet(dim::Integer;  
                            upscaling_dimension = 2 * dim, 
                            n_layers = g_n_layers_default, 
                            activation = g_activation_default, 
                            init_upper = g_init_upper_default)
        new{typeof(activation)}(dim, upscaling_dimension, n_layers, activation, init_upper)
    end

    function ForcedSympNet(dl::DataLoader;   
                                        upscaling_dimension = 2 * dl.input_dim, 
                                        n_layers = g_n_layers_default, 
                                        activation = g_activation_default, 
                                        init_upper = g_init_upper_default) 
        new{typeof(activation)}(dl.input_dim, upscaling_dimension, n_layers, activation, init_upper)
    end
end

function Chain(arch::ForcedSympNet)
    layers = ()
    is_upper_criterion = arch.init_upper ? isodd : iseven
    for i in 1:arch.n_layers
        layers = (layers..., 
        if is_upper_criterion(i)
            GradientLayerQ(arch.dim, arch.upscaling_dimension, arch.act)
        else
            GradientLayerP(arch.dim, arch.upscaling_dimension, arch.act)
        end
        )
        layers = (layers..., ForcingLayerP(arch.dim, arch.upscaling_dimension, arch.n_layers, arch.act; return_parameters=false))
    end
    Chain(layers...)
end