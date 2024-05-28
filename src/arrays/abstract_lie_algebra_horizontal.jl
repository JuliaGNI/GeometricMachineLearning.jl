@doc raw"""
`AbstractLieAlgHorMatrix` is a supertype for various horizontal components of Lie algebras. We usually call this ``\mathfrak{g}^\mathrm{hor}``.

See [`StiefelLieAlgHorMatrix`](@ref) and [`GrassmannLieAlgHorMatrix`](@ref).
"""
abstract type AbstractLieAlgHorMatrix{T} <: AbstractMatrix{T} end
