@doc raw"""
Realizes the linear Symplectic Transformer.

Fields: 
- `dim::Int`: System dimension 
- `time_steps::Int`: Number of time steps that the transformer considers. 
- `depth::Int`: The number of SympNet layers that is applied. 
- `nhidden::Int`: The number of hidden layers in the transformer.
- `upscaling_dimension::Int`: The upscaling that is done by the gradient layer. 
- `activation`: The activation function for the SympNet layers. 
- `init_upper::Bool`: Specifies if the first layer is a ``Q``-type layer (`init_upper=true`) or if it is a ``P``-type layer (`init_upper=false`).
"""
struct LinearSymplecticTransformer{AT, InitUpper} <: Architecture where AT 
    dim::Int
    time_steps::Int
    depth::Int
    nhidden::Int
    upscaling_dimension::Int
    activation::AT
    seq_length::Int

    function LinearSymplecticTransformer(dim::Int, time_steps::Int; depth::Int=5, nhidden::Int=2, upscaling_dimension::Int=2*dim, activation=tanh, init_upper::Bool=true, seq_length::Int=5)
        new{typeof(tanh), init_upper}(dim, time_steps, depth, nhidden, upscaling_dimension, activation, seq_length)
    end

    function LinearSymplecticTransformer(dl::DataLoader{T, BT}; depth::Int=5, nhidden::Int=4, upscaling_dimension::Int=2*dl.input_dim, activation=tanh, init_upper::Bool=true, seq_length::Int=5) where {T, AT<:AbstractArray{T}, BT<:NamedTuple{(:q, :p), Tuple{AT, AT}}}
        new{typeof(tanh), init_upper}(dl.input_dim, dl.input_time_steps, depth, nhidden, upscaling_dimension, activation, seq_length)
    end
end

function Chain(arch::LinearSymplecticTransformer{AT, true}) where AT
    layers = ()
    for _ in 1:(arch.nhidden+1)
        layers = (layers..., LinearSymplecticTransformerLayerQ(arch.seq_length))
        for __ in 1:arch.depth 
            layers = (layers..., GradientLayerQ(arch.dim, arch.upscaling_dimension, arch.activation))
        end
        layers = (layers..., LinearSymplecticTransformerLayerP(arch.seq_length))
        for __ in 1:arch.depth 
            layers = (layers..., GradientLayerP(arch.dim, arch.upscaling_dimension, arch.activation))
        end
    end
    Chain(layers...)
end

function Chain(arch::LinearSymplecticTransformer{AT, false}) where AT
    layers = ()
    for _ in 1:(arch.nhidden+1)
        layers = (layers..., LinearSymplecticTransformerLayerP(arch.seq_length))
        for __ in 1:arch.depth 
            layers = (layers..., GradientLayerP(arch.dim, arch.upscaling_dimension, arch.activation))
        end
        layers = (layers..., LinearSymplecticTransformerLayerQ(arch.seq_length))
        for __ in 1:arch.depth 
            layers = (layers..., GradientLayerQ(arch.dim, arch.upscaling_dimension, arch.activation))
        end
    end
    Chain(layers...)
end