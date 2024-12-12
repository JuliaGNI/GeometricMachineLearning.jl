@doc raw"""
    TransformerIntegrator <: Architecture

Encompasses various transformer architectures, such as the [`VolumePreservingTransformer`](@ref) and the [`LinearSymplecticTransformer`](@ref). 

The central idea behind this is to construct an explicit multi-step integrator:

```math
    \mathtt{Integrator}: [ z^{(t - \mathtt{sl} + 1)}, z^{(t - \mathtt{sl} + 2)}, \ldots, z^{(t)} ] \mapsto [ z^{(t + 1)}, z^{(t + 2)}, \ldots, z^{(t + \mathtt{pw})} ],
```
where `sl` stands for *sequence length* and `pw` stands for *prediction window*, so the numbers of input and output vectors respectively.
"""
abstract type TransformerIntegrator <: Architecture end

@doc raw"""
    DummyTransformer(seq_length)

Make an instance of `DummyTransformer`.

This *dummy architecture* can be used if the user wants to define a new [`TransformerIntegrator`](@ref).
"""
struct DummyTransformer <: TransformerIntegrator 
    seq_length::Int
end

@doc raw"""
    iterate(nn, ics)

Iterate the neural network of type [`TransformerIntegrator`](@ref) for initial conditions `ics`.

The initial condition is a matrix ``\in\mathbb{R}^{n\times\mathtt{seq\_length}}`` or `NamedTuple` of two matrices).

This function computes a trajectory for a Transformer that has already been trained for valuation purposes.

# Parameters 

The following are optional keyword arguments:
- `n_points::Int=100`: The number of time steps for which we run the prediction.
- `prediction_window::Int=size(ics.q, 2)`: The prediction window (i.e. the number of steps we predict into the future) is equal to the sequence length (i.e. the number of input time steps) by default.  
"""
function Base.iterate(nn::NeuralNetwork{<:TransformerIntegrator}, ics::NamedTuple{(:q, :p), Tuple{AT, AT}}; n_points::Int = 100, prediction_window::Union{Nothing, Int}=size(ics.q, 2)) where {T, AT<:AbstractMatrix{T}}

    # if the number of predicted points is zero, just return the initial condition
    if n_points == 0
        return ics
    end

    seq_length = typeof(nn.architecture) <: StandardTransformerIntegrator ? size(ics.q, 2) : nn.architecture.seq_length

    n_dim = size(ics.q, 1)
    backend = networkbackend(ics.q)

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
        q_valuation[:, seq_length + (i - 1) * prediction_window + 1 : seq_length + i * prediction_window] = qp_prediction.q[:, (seq_length - prediction_window + 1):end]
        p_valuation[:, seq_length + (i - 1) * prediction_window + 1 : seq_length + i * prediction_window] = qp_prediction.p[:, (seq_length - prediction_window + 1):end]
    end

    (q=q_valuation[:, 1:n_points], p=p_valuation[:, 1:n_points])
end

function Base.iterate(::NeuralNetwork{<:TransformerIntegrator}, ics::AT; n_points::Int = 100, prediction_window::Union{Nothing, Int} = size(ics, 2)) where {T, AT<:AbstractVector{T}}
    error("You have to provide a matrix as initial condition for the transformer!")
end

function Base.iterate(nn::NeuralNetwork{<:TransformerIntegrator}, ics::AT; n_points::Int = 100, prediction_window::Union{Nothing, Int} = size(ics, 2)) where {T, AT<:AbstractMatrix{T}}
    seq_length = typeof(nn.architecture) <: StandardTransformerIntegrator ? size(ics, 2) : nn.architecture.seq_length

    # if the number of predicted points is zero, just return the initial condition
    if n_points == 0
        return ics
    end

    n_dim = size(ics, 1)
    backend = networkbackend(ics)

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