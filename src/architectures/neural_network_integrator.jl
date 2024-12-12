@doc raw"""
`NeuralNetworkIntegrator` is a super type of various neural network architectures such as [`SympNet`](@ref) and [`ResNet`](@ref).

The purpose of such neural networks is to approximate the flow of an ordinary differential equation (ODE).

`NeuralNetworkIntegrator`s can be seen as modeling traditional one-step methods with neural networks, i.e. for a fixed time step they perform:

```math
    \mathtt{NeuralNetworkIntegrator}: z^{(t)} \mapsto z^{(t+1)},
```

to try to approximate the flow of some ODE:

```math
    || \mathtt{Integrator}(z^{(t)}) - \varphi^h(z^{(t)}) || \approx \mathcal{O}(h),
```

where ``\varphi^h`` is the flow map of the ODE for a time step ``h``.
"""
abstract type NeuralNetworkIntegrator <: Architecture end

function Base.iterate(nn::NeuralNetwork{<:NeuralNetworkIntegrator}, ics::AT; n_points = 100) where {T, AT<:AbstractVector{T}}

    n_dim = length(ics)
    backend = networkbackend(ics)

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
    iterate(nn, ics)

This function computes a trajectory for a [`NeuralNetworkIntegrator`](@ref) that has already been trained for valuation purposes.

It takes as input: 
1. `nn`: a `NeuralNetwork` (that has been trained).
2. `ics`: initial conditions (a `NamedTuple` of two vectors)

# Examples

To demonstrate `iterate` we use a simple ResNet that does:

```math
\mathrm{ResNet}: x \mapsto \begin{pmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 1\end{pmatrix}x + \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}
```

and we iterate three times with

```math
    \mathtt{ics} = \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}.
```

```jldoctest
using GeometricMachineLearning

model = ResNet(3, 0, identity)
weight = [1 0 0; 0 2 0; 0 0 1]
bias = [0, 0, 1]
ps = NeuralNetworkParameters((L1 = (weight = weight, bias = bias), ))
nn = NeuralNetwork(model, Chain(model), ps, CPU())

ics = [1, 1, 1]
iterate(nn, ics; n_points = 4)

# output

3×4 Matrix{Int64}:
 1  2  4   8
 1  3  9  27
 1  3  7  15
```

# Arguments

The optional keyword argument is 
- `n_points = 100`

The number of integration steps that should be performed.
"""
function Base.iterate(nn::NeuralNetwork{<:NeuralNetworkIntegrator}, ics::BT; n_points = 100) where {T, AT<:AbstractVector{T}, BT<:NamedTuple{(:q, :p), Tuple{AT, AT}}}

    n_dim2 = length(ics.q)
    backend = networkbackend(ics.q)

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

@doc raw"""
    DummyNNIntegrator()

Make an instance of `DummyNNIntegrator`.

This *dummy architecture* can be used if the user wants to define a new [`NeuralNetworkIntegrator`](@ref).
"""
struct DummyNNIntegrator <: NeuralNetworkIntegrator end