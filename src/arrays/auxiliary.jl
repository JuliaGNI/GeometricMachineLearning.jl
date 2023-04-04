"""
Auxiliary arrays - these should proably be put into different files or given another name.
"""

struct StiefelProjection{T} <: AbstractMatrix{T}
    N::Int
    n::Int
    StiefelProjection(N, n, T = Float64) = new{T}(N,n)
end
    
function Base.getindex(E::StiefelProjection,i,j)
    if i == j 
        return 1. 
    end 
    return 0. 
end 
   
Base.parent(E::StiefelProjection) = (E.N,E.n)
Base.size(E::StiefelProjection) = (E.N,E.n)