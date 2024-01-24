@doc raw"""
Computes the loss for a neural network and a data set. 
The computed loss is 
```math
||output - \mathcal{NN}(input)||_F/||output||_F,
``` 
where ``||A||_F := \sqrt{\sum_{i_1,\ldots,i_k}|a_{i_1,\ldots,i_k}^2}|^2`` is the Frobenius norm.

It takes as input: 
- `model::Union{Chain, AbstractExplicitLayer}`
- `ps::Union{Tuple, NamedTuple}`
- `input::Union{Array, NamedTuple}`
- `output::Uniont{Array, NamedTuple}`
"""
function loss(model::Union{Chain, AbstractExplicitLayer}, ps::Union{Tuple, NamedTuple}, input::AT, output::BT) where {T, T1, AT<:AbstractArray{T, 3}, BT<:AbstractArray{T1, 3}}
    output_estimate = model(input, ps)
    norm(output - output_estimate) / norm(output) 
end

@doc raw"""
The *autoencoder loss*:
```math
||output - \mathcal{NN}(input)||_F/||output||_F.
```

It takes as input: 
- `model::Union{Chain, AbstractExplicitLayer}`
- `ps::Union{Tuple, NamedTuple}`
- `input::Union{Array, NamedTuple}`
"""
function loss(model::Union{Chain, AbstractExplicitLayer}, ps::Union{Tuple, NamedTuple}, input::BT) where {T, BT<:AbstractArray{T}} 
    output_estimate = model(input, ps)
    norm(output_estimate - input) / norm(input) 
end

nt_diff(A, B) = (q = A.q - B.q, p = A.p - B.p)
nt_norm(A) = norm(A.q) + norm(A.p)

function loss(model::Union{Chain, AbstractExplicitLayer}, ps::Union{Tuple, NamedTuple}, input::NT, output::NT) where {T, AT<:AbstractArray{T}, NT<:NamedTuple{(:q, :p,), Tuple{AT, AT}}}
    output_estimate = model(input, ps)
    nt_norm(nt_diff(output_estimate, output)) / nt_norm(input)
end

function loss(model::Union{Chain, AbstractExplicitLayer}, ps::Union{Tuple, NamedTuple}, input::NT) where {T, AT<:AbstractArray{T}, NT<:NamedTuple{(:q, :p,), Tuple{AT, AT}}}
    output_estimate = model(input, ps)
    nt_norm(nt_diff(output_estimate, input)) / nt_norm(input)
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

You can supply an instance of `NeuralNetwork` instead of the two arguments model (of type `Union{Chain, AbstractExplicitLayer}`) and parameters (of type `Union{Tuple, NamedTuple}`).
"""
function loss(nn::NeuralNetwork, var_args...)
    loss(nn.model, nn.params, var_args...)
end