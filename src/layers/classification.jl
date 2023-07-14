"""
Classification layer that takes as input a vector, a matrix or a tensor (input is called x) and outputs a vector in the first two cases and a matrix in the second case. 

In the case it takes a vector or a matrix as input, the output is a vector of dimension n_labels. In the case it takes a tensor, the output is a matrix with dimensions n_labels and the batch size. 

Optional keywords are: 
- use_bias
- use_average (for matrix and tensor inputs): if true this computes sum(ps.weight*x, dims=2)/size(x, 2) and if false this computes ps.weight*x[:, 1, :] (or ps.weight*x[:, 1] in the matrix case) before the nonlinearity is employed.
- use_softmax: if true the nonlinearity is softmax (used column-wise). if false then it is a componentwise-employed sigmoid. 
"""
struct Classification{use_bias, use_average, use_softmax} <: Lux.AbstractExplicitLayer
    n_labels::Integer 
    input_dim::Integer

    function Classification(n_labels::Integer, input_dim::Integer; use_bias::Bool=true, use_average::Bool=true, use_softmax=true)
        new{use_bias, use_average, use_softmax}(n_labels, input_dim)
    end
end

function Lux.initialparameters(rng::Random.AbstractRNG, d::Classification{use_bias}) where use_bias
    use_bias ? (weight=Lux.glorot_uniform(rng, d.input_dim, d.n_labels), bias=zeros(Float32, d.n_labels)) : 
        (weight=Lux.glorot_uniform(rng, d.input_dim, d.n_labels), )
end

function Lux.parameterlength(d::Classification{use_bias}) where {use_bias}
    return use_bias ? d.n_labels * d.input_dim : d.n_labels * d.input_dim
end

function Lux.apply(d::Classification{true, use_average, true}, x::AbstractVector, ps::NamedTuple, st::NamedTuple) where use_average
    Lux.softmax(ps.weight*x + ps.bias), st
end

function Lux.apply(d::Classification{true, use_average, false}, x::AbstractVector, ps::NamedTuple, st::NamedTuple) where use_average
    Lux.sigmoid.(ps.weight*x + ps.bias), st
end

function Lux.apply(d::Classification{false, use_average, true}, x::AbstractVector, ps::NamedTuple, st::NamedTuple) where use_average
    Lux.softmax(ps.weight*x), st
end

function Lux.apply(d::Classification{false, use_average, false}, x::AbstractVector, ps::NamedTuple, st::NamedTuple) where use_average
    Lux.sigmoid.(ps.weight*x), st
end

function Lux.apply(d::Classification{true, true, true}, x::AbstractMatrix, ps::NamedTuple, st::NamedTuple)
    Lux.softmax(sum(ps.weight*x, dims=2)[:, 1]/size(x, 2) + ps.bias), st
end

function Lux.apply(d::Classification{true, true, false}, x::AbstractMatrix, ps::NamedTuple, st::NamedTuple)
    Lux.sigmoid.(sum(ps.weight*x, dims=2)[:, 1]/size(x, 2) + ps.bias), st
end

function Lux.apply(d::Classification{true, false, true}, x::AbstractMatrix, ps::NamedTuple, st::NamedTuple)
    Lux.softmax(ps.weight*(x[:,1]) + ps.bias), st
end

function Lux.apply(d::Classification{true, false, false}, x::AbstractMatrix, ps::NamedTuple, st::NamedTuple)
    Lux.sigmoid.(ps.weight*(x[:,1]) + ps.bias), st
end

function Lux.apply(d::Classification{false, false, true}, x::AbstractMatrix, ps::NamedTuple, st::NamedTuple)
    Lux.softmax(ps.weight*(x[:,1])), st
end

function Lux.apply(d::Classification{false, false, false}, x::AbstractMatrix, ps::NamedTuple, st::NamedTuple)
    Lux.sigmoid.(ps.weight*(x[:,1])), st
end

function Lux.apply(d::Classification{true, true, true}, x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where T
    Lux.softmax(sum(mat_tensor_mul(ps.weight, x), dims=2)[:,1,:]/size(x, 2) .+ ps.bias), st
end

function Lux.apply(d::Classification{true, true, false}, x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where T
    Lux.sigmoid.(sum(mat_tensor_mul(ps.weight, x), dims=2)[:,1,:]/size(x, 2) .+ ps.bias), st
end

function Lux.apply(d::Classification{false, true, true}, x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where T
    Lux.softmax(sum(mat_tensor_mul(ps.weight, x), dims=2)[:,1,:]/size(x, 2)), st
end

function Lux.apply(d::Classification{false, true, false}, x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where T
    Lux.sigmoid.(sum(mat_tensor_mul(ps.weight, x), dims=2)[:,1,:]/size(x, 2)), st
end

function Lux.apply(d::Classification{true, false, true}, x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where T
    Lux.softmax(ps.weight*x[:,1,:] .+ ps.bias), st
end

function Lux.apply(d::Classification{true, false, false}, x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where T
    Lux.sigmoid.(ps.weight*x[:,1,:] .+ ps.bias), st
end

function Lux.apply(d::Classification{false, false, true}, x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where T
    Lux.softmax(ps.weight*x[:,1,:]), st
end

function Lux.apply(d::Classification{false, false, false}, x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where T
    Lux.sigmoid.(ps.weight*x[:,1,:]), st
end

(d::Classification)(x::AbstractArray, ps::NamedTuple, st::NamedTuple) = Lux.apply(d, x, ps, st)