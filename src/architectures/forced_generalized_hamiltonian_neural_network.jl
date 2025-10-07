"""
    ForcedGeneralizedHamiltonianArchitecture <: HamiltonianArchitecture

A version of [`GeneralizedHamiltonianArchitecture`](@ref) that includes forcing/dissipation terms. Also compare this to [`ForcedSympNet`](@ref).
"""
struct ForcedGeneralizedHamiltonianArchitecture{FT, AT, PT <: OptionalParameters} <: HamiltonianArchitecture{AT}
    dim::Int
    width::Int
    nhidden::Int
    n_integrators::Int
    parameters::PT
    activation::AT

    function ForcedGeneralizedHamiltonianArchitecture(dim; width=dim, nhidden=HNN_nhidden_default, n_integrators::Integer=1, activation=HNN_activation_default, parameters=NullParameters(), forcing_type::Symbol=:P)
        forcing_type == :P || forcing_type == :Q || error("Forcing has to be either :Q or :P. It is $(forcing_type).")
        activation = (typeof(activation) <: Activation) ? activation : Activation(activation)
        new{forcing_type, typeof(activation), typeof(parameters)}(dim, width, nhidden, n_integrators, parameters, activation)
    end
end

function Chain(arch::ForcedGeneralizedHamiltonianArchitecture{FT}) where {FT}
    layers = ()
    kinetic_energy = SymbolicKineticEnergy(arch.dim, arch.width, arch.nhidden, arch.activation; parameters=arch.parameters)
    potential_energy = SymbolicPotentialEnergy(arch.dim, arch.width, arch.nhidden, arch.activation; parameters=arch.parameters)
    for i ∈ 1:arch.n_integrators
        layers = (layers..., SymplecticEulerA(kinetic_energy; return_parameters = true))
        layers = (layers..., ForcingLayer(arch.dim, arch.width, arch.nhidden, arch.activation; parameters=arch.parameters, return_parameters=true, type=FT))
        _return_parameters = !(i == arch.n_integrators)
        layers = (layers..., SymplecticEulerB(potential_energy; return_parameters = _return_parameters))
    end
    Chain(layers...)
end