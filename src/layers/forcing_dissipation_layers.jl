@doc raw"""
    ForcingLayer <: AbstractExplicitLayer

An abstract type that summarizes layers that can learn dissipative terms, but not conservative ones. 
"""
abstract type ForcingLayer{M, N} <: AbstractExplicitLayer{M, N} end