# AbstractRetraction is a type that comprises all retraction methods for manifolds. For every manifold layer one has to specify a retraction method that takes the layer and elements of the (global) tangent space.
abstract type AbstractRetraction end 

struct Cayley <: AbstractRetraction end

struct Geodesic <: AbstractRetraction end