struct Classification{use_bias， use_average} <: Lux.AbstractExplicitLayer
    n_labels::Integer 
    input_dim::Integer

    function Classification(n_labels::Integer, input_dim::Integer; use_bias::Bool=true, use_average::Bool=true)
        new{use_bias, use_average}(n_labels, input_dim)
    end
end

function Lux.initialparameters(rng::Random.AbstractRNG, d::Classification{use_bias}) where use_bias
    use_bias ? (weight=Lux.glorot_uniform(rng, d.input_dim, d.n_labels), bias=zeros(Float32, d.n_labels)) : 
        (weight=Lux.glorot_uniform(rng, d.input_dim, d.n_labels), )
end

function Lux.parameterlength(d::Classification{use_bias}) where {use_bias}
    return use_bias ? d.n_labels * d.input_dim : d.n_labels * d.input_dim
end

function Lux.apply(d::Classification{true}, x::AbstractVector, ps::NamedTuple, st::NamedTuple)
    Lux.softmax(ps.weight*x + ps.bias), st
end

function Lux.apply(d::Classification{false}, x::AbstractVector, ps::NamedTuple, st::NamedTuple)
    Lux.softmax(ps.weight*x), st
end

function Lux.apply(d::Classification{true, true}, x::AbstractMatrix, ps::NamedTuple, st::NamedTuple)
    Lux.softmax(sum(ps.weight*x, dims=2)[:, 1]/size(x, 2) + ps.bias), st
end

function Lux.apply(d::Classification{true, false}, x::AbstractMatrix, ps::NamedTuple, st::NamedTuple)
    Lux.softmax(ps.weight*(x[:,1]) + ps.bias), st
end

function Lux.apply(d::Classification{false, false}, x::AbstractMatrix, ps::NamedTuple, st::NamedTuple)
    Lux.softmax(ps.weight*(x[:,1])), st
end

function Lux.apply(d::Classification{true, true}, x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where T
    Lux.softmax(sum(mat_tensor_mul(ps.weight, x), dims=2)[:,1,:]/size(x, 2) .+ ps.bias), st
end

function Lux.apply(d::Classification{false, true}, x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where T
    Lux.softmax(sum(mat_tensor_mul(ps.weight, x), dims=2)[:,1,:]/size(x, 2)), st
end

function Lux.apply(d::Classification{true, false}, x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where T
    Lux.softmax(mat_tensor_mul(ps.weight, x[:,1,:]) .+ ps.bias), st
end

function Lux.apply(d::Classification{false, false}, x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where T
    Lux.softmax(mat_tensor_mul(ps.weight, x[:,1,:])), st
end

(d::Classification)(x::AbstractArray, ps::NamedTuple, st::NamedTuple) = Lux.apply(d, x, ps, st)