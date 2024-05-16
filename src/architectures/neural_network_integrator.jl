@doc raw"""
This is a super type of various neural network architectures such as [`SympNet`](@ref) and [`ResNet`](@ref) whose purpose is to approximate the flow of an ordinary differential equation (ODE).
"""
abstract type NeuralNetworkIntegrator <: Architecture end

function Base.iterate(nn::NeuralNetwork{<:NeuralNetworkIntegrator}, ics::AT; n_points = 100) where {T, AT<:AbstractVector{T}}

    n_dim = length(ics)
    backend = KernelAbstractions.get_backend(ics)

    # Array to store the predictions
    valuation = KernelAbstractions.allocate(backend, T, n_dim, n_points)
    
    # Initialisation
    @views valuation[:,1] = ics
    
    #Computation of phase space
    @views for i in 2:n_points
        temp = valuation[:,i-1]
        prediction = nn(temp)
        valuation[:,i] = prediction
    end

    valuation
end

function Base.iterate(nn::NeuralNetwork{<:NeuralNetworkIntegrator}, ics::BT; n_points = 100) where {AT<:AbstractVector, BT<:NamedTuple{(:q, ), Tuple{AT}}}
    (q = iterate(nn, ics.q; n_points = n_points), )
end

@doc raw"""
This function computes a trajectory for a SympNet that has already been trained for valuation purposes.

It takes as input: 
- `nn`: a `NeuralNetwork` (that has been trained).
- `ics`: initial conditions (a `NamedTuple` of two vectors)
"""
function Base.iterate(nn::NeuralNetwork{<:NeuralNetworkIntegrator}, ics::BT; n_points = 100) where {T, AT<:AbstractVector{T}, BT<:NamedTuple{(:q, :p), Tuple{AT, AT}}}

    n_dim2 = length(ics.q)
    backend = KernelAbstractions.get_backend(ics.q)

    # Array to store the predictions
    valuation = (q = KernelAbstractions.allocate(backend, T, n_dim2, n_points), p = KernelAbstractions.allocate(backend, T, n_dim2, n_points))
    
    # Initialisation
    @views valuation.q[:, 1] = ics.q 
    @views valuation.p[:, 1] = ics.p
    
    #Computation of phase space
    @views for i in 2:n_points
        temp = (q = valuation.q[:, i-1], p = valuation.p[:, i-1])
        prediction = nn(temp)
        valuation.q[:, i] = prediction.q 
        valuation.p[:, i] = prediction.p
    end

    valuation
end

struct DummyNNIntegrator <: NeuralNetworkIntegrator end