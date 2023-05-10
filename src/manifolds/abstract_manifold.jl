#abstract type Manifold <: AbstractMatrix end

Manifold = Union{SymplecticStiefelManifold, StiefelManifold}

Base.size(A::Manifold) = size(A.A)
Base.parent(A::Manifold) = A.A 
Base.getindex(A::Manifold, i::Int, j::Int) = A.A[i,j]