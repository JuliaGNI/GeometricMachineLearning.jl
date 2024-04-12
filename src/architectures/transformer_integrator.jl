@doc raw"""
Encompasses various transformer architectures, such as the structure-preserving transformer and the linear symplectic transformer. 
"""
abstract type TransformerIntegrator <: Architecture end

struct DummyTransformer <: TransformerIntegrator 
    seq_length::Int
end

@doc raw"""
This function computes a trajectory for a Transformer that has already been trained for valuation purposes.

It takes as input: 
- `nn`: a `NeuralNetwork` (that has been trained).
- `ics`: initial conditions (a matrix in ``\mathbb{R}^{2n\times\mathtt{seq\_length}}`` or `NamedTuple` of two matrices in ``\mathbb{R}^{n\times\mathtt{seq\_length}}``)
- `n_points::Int=100` (keyword argument): The number of steps for which we run the prediction. 
"""
function Base.iterate(nn::NeuralNetwork{<:TransformerIntegrator}, ics::NamedTuple{(:q, :p), Tuple{AT, AT}}; n_points::Int = 100, prediction_window::Union{Nothing, Int} = 1) where {T, AT<:AbstractMatrix{T}}

    seq_length = nn.architecture.seq_length

    n_dim = size(ics.q, 1)
    backend = KernelAbstractions.get_backend(ics.q)

    n_iterations = Int(ceil((n_points - seq_length) / prediction_window))
    # Array to store the predictions
    q_valuation = KernelAbstractions.allocate(backend, T, n_dim, seq_length + n_iterations * prediction_window)
    p_valuation = KernelAbstractions.allocate(backend, T, n_dim, seq_length + n_iterations * prediction_window)
    
    # Initialisation
    q_valuation[:,1:seq_length] = ics.q
    p_valuation[:,1:seq_length] = ics.p
    
    # iteration in phase space
    @views for i in 1:n_iterations
        start_index = (i - 1) * prediction_window + 1
        @views qp_temp = (q = q_valuation[:, start_index:(start_index + seq_length - 1)], p = p_valuation[:, start_index:(start_index + seq_length - 1)]) 
        qp_prediction = nn(qp_temp)
        q_valuation[seq_length + (i - 1) * prediction_window, seq_length + i * prediction_window] = qp_prediction.q[:, (seq_length - prediction_window + 1):end]
        p_valuation[seq_length + (i - 1) * prediction_window, seq_length + i * prediction_window] = qp_prediction.p[:, (seq_length - prediction_window + 1):end]
    end

    (q=q_valuation[:, 1:n_points], p=p_valuation[:, 1:n_points])
end

function Base.iterate(nn::NeuralNetwork{<:TransformerIntegrator}, ics::AT; n_points::Int = 100, prediction_window::Union{Nothing, Int} = 1) where {T, AT<:AbstractMatrix{T}}

    seq_length = typeof(nn.architecture) <: RegularTransformerIntegrator ? prediction_window : nn.architecture.seq_length

    n_dim = size(ics, 1)
    backend = KernelAbstractions.get_backend(ics)

    n_iterations = Int(ceil((n_points - seq_length) / prediction_window))
    # Array to store the predictions
    valuation = KernelAbstractions.allocate(backend, T, n_dim, seq_length + n_iterations * prediction_window)
    
    # Initialisation
    valuation[:,1:seq_length] = ics
    
    # iteration in phase space
    @views for i in 1:n_iterations
        start_index = (i - 1) * prediction_window + 1
        temp = valuation[:, start_index:(start_index + seq_length - 1)]
        prediction = nn(copy(temp))
        valuation[:, (seq_length + (i - 1) * prediction_window + 1):(seq_length + i * prediction_window)] = prediction[:, (seq_length - prediction_window + 1):end]
    end

    valuation[:, 1:n_points]
end