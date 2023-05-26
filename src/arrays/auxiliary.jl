"""
Auxiliary arrays - these should proably be put into different files or given another name.
"""

struct StiefelProjection{T} <: AbstractMatrix{T}
    N::Integer
    n::Integer
    StiefelProjection(N, n, T=Float64) = new{T}(N,n)
end
    
function Base.getindex(::StiefelProjection{T},i,j) where T
    if i == j 
        return T(1.) 
    end 
    return T(0.) 
end 
   
Base.parent(E::StiefelProjection) = (E.N, E.n)

Base.size(E::StiefelProjection) = (E.N, E.n)

struct One{T} <: AbstractMatrix{T}
    n::Integer
    One(n, T=Float64) = new{T}(n)
end

function Base.getindex(::One{T}, i, j) where T
if i == j 
    return T(1.) 
    end 
    return T(0.) 
end 
Base.parent(mat::One) = (mat.n)
Base.size(mat::One) = (mat.n, mat.n)

