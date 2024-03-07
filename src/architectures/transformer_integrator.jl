@doc raw"""
Encompasses various transformer architectures, such as the structure-preserving transformer and the linear symplectic transformer. 
"""
abstract type TransformerIntegrator <: Architecture end

@doc raw"""
This function computes a trajectory for a Transformer that has already been trained for valuation purposes.

It takes as input: 
- `nn`: a `NeuralNetwork` (that has been trained).
- `ics`: initial conditions (a matrix in ``\mathbb{R}^{2n\times\mathtt{seq\_length}}`` or `NamedTuple` of two matrices in ``\mathbb{R}^{n\times\mathtt{seq\_length}}``)
- `n_points::Int=100` (keyword argument): The number of steps for which we run the prediction. 
"""
function Base.iterate(nn::NeuralNetwork{<:TransformerIntegrator}, ics::NamedTuple{(:q, :p), Tuple{AT, AT}}; n_points::Int = 100) where {T, AT<:AbstractMatrix{T}}

    seq_length = nn.model.seq_length
    n_dim = size(ics.q, 1)
    backend = KernelAbstractions.get_backend(ics.q)

    # Array to store the predictions
    q_valuation = KernelAbstractions.allocate(backend, T, n_dim, n_points)
    p_valuation = KernelAbstractions.allocate(backend, T, n_dim, n_points)
    
    # Initialisation
    q_valuation[:,1:seq_length] = ics.q
    p_valuation[:,1:seq_length] = ics.p
    
    # iteration in phase space
    @views for i in (seq_length + 1):n_points
        qp_temp = (q=q_valuation[:, (i - seq_length):(i - 1)], p=p_valuation[:, (i - seq_length):(i - 1)]) 
        qp_prediction = nn(qp_temp)
        q_valuation[:, i] = qp_prediction.q[:, end]
        p_valuation[:, i] = qp_prediction.p[:, end]
    end

    (q=q_valuation, p=p_valuation)
end

function Base.iterate(nn::NeuralNetwork{<:TransformerIntegrator}, ics::AT; n_points::Int = 100, seq_length::Union{Nothing, Int} = nothing) where {T, AT<:AbstractMatrix{T}}

    seq_length = isnothing(seq_length) ? nn.architecture.seq_length : seq_length
    @assert size(ics, 2) == seq_length

    n_dim = size(ics, 1)
    backend = KernelAbstractions.get_backend(ics)

    # Array to store the predictions
    valuation = KernelAbstractions.allocate(backend, T, n_dim, n_points)
    
    # Initialisation
    valuation[:,1:seq_length] = ics
    
    # iteration in phase space
    @views for i in (seq_length + 1):n_points
        temp = valuation[:, (i - seq_length):(i - 1)] 
        prediction = nn(temp)
        valuation[:, i] = prediction[:, end]
    end

    valuation
end

struct DummyTransformer <: TransformerIntegrator 
    seq_length::Int
end