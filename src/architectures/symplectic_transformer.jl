@doc raw"""
Realizes the Symplectic Transformer.

Fields: 
- `dim::Int`: System dimension 
- `time_steps::Int`: Number of time steps that the transformer considers. 
- `depth::Int`: The number of SympNet layers that is applied. 
- `upscaling_dimension::Int`: The upscaling that is done by the gradient layer. 
- `activation`: The activation function for the SympNet layers. 
- `init_upper::Bool`: Specifies if the first layer is a ``Q``-type layer (`init_upper=true`) or if it is a ``P``-type layer (`init_upper=false`).
"""
struct SymplecticTransformer{AT, InitUpper} <: Architecture where AT 
    dim::Int
    time_steps::Int
    depth::Int
    nhidden::Int
    activation::AT
    upscaling_dimension::AT

    function SymplecticTransformer(dim::Int, time_steps::Int; depth::Int=5, nhidden::Int=2, activation=tanh, init_upper::Bool=true)
        new{typeof(AT), init_upper}(dim, time_steps, depth, nhidden, activation)
    end

    function SymplecticTransformer(dl::DataLoader; depth::Int=5, nhidden::Int=4, activation=tanh, init_upper::Bool=true)
        new{typeof(AT), init_upper}(dl.input_dim, dl.input_time_steps, depth, nhidden, activation)
    end
end

function Chain(arch::SymplecticTransformer{AT, true}) where AT
    layers = ()
    for _ in 1:(arch.nhidden+1)
        layers = (layers..., SymplecticTransformerLayerQ(arch.dim))
        for ⦸ in 1:arch.depth 
            layers = (layers..., GradientLayerQ(arch.dim, arch.upscaling_dimension, arch.act))
        end
        layers = (layers..., SymplecticTransformerLayerP(arch.dim))
        for ⦸ in 1:arch.depth 
            layers = (layers..., GradientLayerP(arch.dim, arch.upscaling_dimension, arch.act))
        end
    end
    Chain(layers...)
end