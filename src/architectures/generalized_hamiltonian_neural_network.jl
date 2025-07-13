"""
    SymbolicEnergy

See [`SymbolicPotentialEnergy`](@ref) and [`SymbolicKineticEnergy`](@ref).
"""
struct SymbolicEnergy{AT <: Activation, PT <: OptionalParameters, Kinetic} 
    dim::Int
    width::Int
    nhidden::Int
    activation::AT
    parameters::PT

    function SymbolicEnergy{Kinetic}(dim, width=dim÷2, nhidden=HNN_nhidden_default, activation=HNN_activation_default; parameters::PT=NullParameters()) where {Kinetic, PT}
        @assert iseven(dim) "The input dimension must be an even integer!"
        new{typeof(activation), PT, Kinetic}(dim ÷ 2, width, nhidden, activation, parameters)
    end
end

"""
    SymbolicPotentialEnergy

# Constructors

```julia
SymbolicPotentialEnergy(dim)
```

# Parameter Dependence
"""
const SymbolicPotentialEnergy{AT} = SymbolicEnergy{AT, :potential}

"""
    SymbolicKineticEnergy

# Constructors

```julia

```
"""
const SymbolicKineticEnergy{AT} = SymbolicEnergy{AT, :kinetic}

SymbolicPotentialEnergy(args...; kwargs...) = SymbolicEnergy{:potential}(args...; kwargs...)
SymbolicKineticEnergy(args...; kwargs...) = KineticEnergy{:kinetic}(args...; kwargs...)

GHNN_integrator_default = nothing

"""
    GeneralizedHamiltonianArchitecture <: HamiltonianArchitecture

A realization of generalized Hamiltonian neural networks (GHNNs) as introduced in [horn2025generalized](@cite).

Also see [`StandardHamiltonianArchitecture`](@ref).

# Constructor

The constructor takes the following input arguments:
1. `dim`: system dimension,
2. `width = dim`: width of the hidden layer. By default this is equal to `dim`,
3. `nhidden = $(HNN_nhidden_default)`: the number of hidden layers,
4. `activation = $(HNN_activation_default)`: the activation function used in the GHNN,
5. `integrator = $(GHNN_integrator_default)`: the integrator that is used to design the GHNN.
"""
struct GeneralizedHamiltonianArchitecture{AT, IT} <: HamiltonianArchitecture{AT}
    dim::Int
    width::Int
    nhidden::Int
    activation::AT
    integrator::IT

    function GeneralizedHamiltonianArchitecture(dim, width=dim, nhidden=HNN_nhidden_default, activation=HNN_activation_default, integrator=GHNN_integrator_default)
        error("GHNN still has to be implemented!")
        new{typeof(activation), typeof(integrator)}(dim, width, nhidden, activation, integrator)
    end
end