using Lux
using Random
using Manifolds

struct SympSt{inverse,F1} <: Lux.AbstractExplicitLayer
    dim_in::Int
    dim_out::Int
    manifold::Manifolds.SymplecticStiefel
    initial_weight::F1
end

#maybe implement another random number generator
function SympSt(dim_in::Int, dim_out::Int;inverse::Bool=false)
    M = Manifolds.SymplecticStiefel(dim_out,dim_in)    
    init_weight = () -> rand(M,1)
    return SymSt{inverse,typeof(init_weight)}(dim_in,dim_out,M,init_weight)
end

function Lux.initialparameters(d::SympSt)
    return (weight=d.init_weight(),)
end

Lux.initialstates(d::SympSt) = NamedTuple()

function Lux.parameterlength(d::SympSt)
        return (4*d.dim_out - 2*d.dim_in + 1)*d.dim_in
end
Lux.statelength(d::SympSt) = 0

@inline function(d::SympSt{false})(x::AbstractVecOrMat, ps, st::NamedTuple)
    return ps.weight*x
end

#maybe pick the symplectic inverse here
@inline function(d::SympSt{true})(x::AbstractVecOrMat, ps, st::NamedTuple)
    return ps.weight'*x
end 

