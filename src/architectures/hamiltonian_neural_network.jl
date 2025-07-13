"""
    HamiltonianArchitecture <: Architecture

See [`StandardHamiltonianArchitecture`](@ref) and [`GeneralizedHamiltonianArchitecture`](@ref).
"""
abstract type HamiltonianArchitecture{AT<:Activation} <: Architecture end

const HNN_nhidden_default = 1
const HNN_activation_default = AbstractNeuralNetworks.TanhActivation()

function HamiltonianArchitecture(dim::Integer, width::Integer, nhidden::Integer, activation)
    @warn "You called the abstract type `HamiltonianArchitecture` as a constructor. This is defaulting to `StandardHamiltonianArchitecture`."
    StandardHamiltonianArchitecture(dim, width, nhidden, activation)
end
