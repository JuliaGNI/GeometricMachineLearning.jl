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

@doc raw"""
This function computes a trajectory for a LinearSymplecticTransformer that has already been trained for valuation purposes.

It takes as input: 
- `nn`: a `NeuralNetwork` (that has been trained).
- `ics`: initial conditions (a `NamedTuple` of two matrices)
"""
function iterate(nn::NeuralNetwork{<:LinearSymplecticTransformer}, ics::NamedTuple{(:q, :p), Tuple{AT, AT}}; n_points = 100) where {T, AT<:AbstractVector{T}}

    seq_length = nn.model.seq_length
    n_dim = length(ics.q)
    backend = KernelAbstractions.get_backend(ics.q)

    # Array to store the predictions
    q_valuation = KernelAbstractions.allocate(backend, T, n_dim, n_points)
    p_valuation = KernelAbstractions.allocate(backend, T, n_dim, n_points)
    
    # Initialisation
    @views q_valuation[:,1:seq_length] = ics.q
    @views p_valuation[:,1:seq_length] = ics.p
    
    #Computation of phase space
    @views for i in (seq_length + 1):n_points
        qp_temp = (q=q_valuation[:, (i - seq_length):(i - 1)], p=p_valuation[:, (i - seq_length):(i - 1)]) 
        qp_prediction = nn(qp_temp)
        q_valuation[:, i] = qp_prediction.q
        p_valuation[:, i] = qp_prediction.p
    end

    (q=q_valuation, p=p_valuation)
end