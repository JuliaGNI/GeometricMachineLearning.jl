@doc raw"""
    AbstractLieAlgHorMatrix <: AbstractMatrix

`AbstractLieAlgHorMatrix` is a supertype for various horizontal components of Lie algebras. We usually call this ``\mathfrak{g}^\mathrm{hor}``.

See [`StiefelLieAlgHorMatrix`](@ref) and [`GrassmannLieAlgHorMatrix`](@ref) for concrete examples.
"""
abstract type AbstractLieAlgHorMatrix{T} <: AbstractMatrix{T} end
