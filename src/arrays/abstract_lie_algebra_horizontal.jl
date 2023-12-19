@doc raw"""
`AbstractLieAlgHorMatrix` is a supertype for various horizontal components of Lie algebras. We usually call this \(\mathfrak{g}^\mathrm{hor}\).
"""
abstract type AbstractLieAlgHorMatrix{T} <: AbstractMatrix{T} end
