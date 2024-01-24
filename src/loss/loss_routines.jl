@doc raw"""
Computes the loss for a neural network and a data set. 
The computed loss is $||output - \mathcal{NN}(input)||_F/\mathtt{size(output, 2)}/\mathtt{size(output, 3)}$, where $||A||_F := \sqrt{\sum_{i_1,\ldots,i_k}||a_{i_1,\ldots,i_k}^2}$ is the Frobenius norm.

It takes as input: 
- `model`
- `ps`
- `input`
- `output`
"""
function loss(model::Union{Chain, AbstractExplicitLayer}, ps::Union{Tuple, NamedTuple}, input::AT, output::BT) where {T, T1, AT<:AbstractArray{T, 3}, BT<:AbstractArray{T1, 3}}
    output_estimate = model(input, ps)
    norm(output - output_estimate) / norm(output) # /T(sqrt(size(output, 2)*size(output, 3)))
end

@doc raw"""
The *autoencoder loss*. 
"""
function loss(model::Chain, ps::Tuple, input::BT) where {T, BT<:AbstractArray{T}} 
    output_estimate = model(input, ps)
    norm(output_estimate - input) / norm(input) # /T(sqrt(size(input, 2)*size(input, 3)))
end

nt_diff(A, B) = (q = A.q - B.q, p = A.p - B.p)
nt_norm(A) = norm(A.q) + norm(A.p)

function loss(model::Chain, ps::Tuple, input::NT) where {T, AT<:AbstractArray{T}, NT<:NamedTuple{(:q, :p,), Tuple{AT, AT}}}
    output_estimate = model(input, ps)
    nt_norm(nt_diff(output_estimate, input)) / nt_norm(input)
end

@doc raw"""
Loss function that takes a `NamedTuple` as input. This should be used with a SympNet (or other neural network-based integrator). It computes:

```math
\mathtt{loss}(\mathcal{NN}, \mathtt{ps}, \begin{pmatrix} q \\ p \end{pmatrix}, \begin{pmatrix} q' \\ p' \end{pmatrix}) \mapsto \left|| \mathcal{NN}(\begin{pmatrix} q \\ p \end{pmatrix}) -  \begin{pmatrix} q' \\ p' \end{pmatrix} \right|| / \left|| \begin{pmatrix} q \\ p \end{pmatrix} \right||
```
"""
function loss(model::Chain, ps::Tuple, input::NamedTuple, output::NamedTuple) 
    output_estimate = model(input, ps)
    nt_norm(nt_diff(output_estimate, output)) / nt_norm(input)
end

@doc raw"""
Alternative call of the loss function. This takes as input: 
- `model`
- `ps`
- `dl::DataLoader`
"""
function loss(model::Union{Chain, AbstractExplicitLayer}, ps::Union{Tuple, NamedTuple}, dl::DataLoader{T, AT, BT}) where {T, T1, AT<:AbstractArray{T, 3}, BT<:AbstractArray{T1, 3}}
    loss(model, ps, dl.input, dl.output)
end

function loss(model::Chain, ps::Tuple, dl::DataLoader{T, BT, Nothing}) where {T, BT<:AbstractArray{T, 3}} 
    loss(model, ps, dl.input)
end

function loss(model::Chain, ps::Tuple, dl::DataLoader{T, BT, Nothing}) where {T, BT<:AbstractArray{T, 2}} 
    loss(model, ps, dl.input)
end

function loss(model::Chain, ps::Tuple, dl::DataLoader{T, BT}) where {T, BT<:NamedTuple}
    loss(model, ps, dl.input)
end

@doc raw"""
Wrapper if we deal with a neural network.
"""
function loss(nn::NeuralNetwork, dl::DataLoader)
    loss(nn.model, nn.params, dl)
end