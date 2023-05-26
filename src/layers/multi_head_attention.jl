struct MultiHeadAttention{Stiefel, Retraction, F1} <: Lux.AbstractExplicitLayer
    dim::Integer
    n_heads::Integer
    init_weight::F1
end

default_retr = Geodesic()
function MultiHeadAttention(dim::Integer, n_heads::Integer; Stiefel::Bool=false, Retraction::AbstractRetraction=default_retr, init_weight=Lux.glorot_uniform)
    @assert dim % n_heads == 0
    MultiHeadAttention{Stiefel, typeof(Retraction), typeof(init_weight)}(dim, n_heads, init_weight)
end

Lux.parameterlength(d::StiefelLayer{false}) = 3*d.dim^2
Lux.parameterlength(d::StiefelLayer{true}) = 3*d.dim*(2*d.dim*d.n_heads - d.n_heads - d.dim)÷(2*d.n_heads^2)


function Lux.initialparameters(rng::AbstractRNG, d::MultiHeadAttention{false})
    #number of "hidden" dimension (dimension of projection) 
    Dₕ = d.dim ÷ d.n_heads
    #projections for queries, keys and vectors.
    PQ = NamedTuple()
    PK = NamedTuple()
    PV = NamedTuple()

    for head in 1:d.n_heads
        key = Symbol("head_"*string(head))
        
        PQ = merge(PQ, 
            NamedTuple{(key, )}((d.init_weight(rng, d.dim, Dₕ), ))
            )
        PK = merge(PK, 
            NamedTuple{(key, )}((d.init_weight(rng, d.dim, Dₕ), ))
            )
        PV = merge(PV, 
            NamedTuple{(key, )}((d.init_weight(rng, d.dim, Dₕ), ))
            )
    end
    (PQ=PQ, PK=PK, PV=PV)
end


function Lux.initialparameters(rng::AbstractRNG, d::MultiHeadAttention{true})
    #number of "hidden" dimension (dimension of projection) 
    Dₕ = d.dim ÷ d.n_heads
    #projections for queries, keys and vectors.
    PQ = NamedTuple()
    PK = NamedTuple()
    PV = NamedTuple()

    for head in 1:d.n_heads
        key = Symbol("head_"*string(head))
        
        PQ = merge(PQ, 
            NamedTuple{(key, )}((d.init_weight(rng, StiefelManifold, d.dim, Dₕ), ))
            )
        PK = merge(PK, 
            NamedTuple{(key, )}((d.init_weight(rng, StiefelManifold, d.dim, Dₕ), ))
            )
        PV = merge(PV, 
            NamedTuple{(key, )}((d.init_weight(rng, StiefelManifold, d.dim, Dₕ), ))
            )
    end
    (PQ=PQ, PK=PK, PV=PV)
end

function Lux.apply(d::MultiHeadAttention, x::AbstractVecOrMat, ps::NamedTuple, st::NamedTuple)
    Dₕ = d.dim ÷ d.n_heads
    dim, input_length = size(x)
    @assert dim == d.dim

    output = zeros(0, input_length)
    for i in 1:d.n_heads
        key = Symbol("head_"*string(i))
        output = vcat(output, ps.PV[key]'*x*Lux.softmax((ps.PQ[key]'*x)'*(ps.PK[key]'*x)))
    end
    x + output, st
end
