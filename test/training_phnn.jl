using GeometricMachineLearning
using Test
using GeometricProblems.CoupledHarmonicOscillator: hodeensemble, default_parameters
using GeometricIntegrators: ImplicitMidpoint, integrate
using Random: seed!
seed!(123)

function make_alternative_parameters_by_adding_constant(params::NamedTuple=default_parameters, n::Integer=1, a::Number=1.)
    _keys = keys(params)
    values = ()
    for (key, i) in zip(_keys, 1:length(_keys))
        values = i == n ? (values..., params[key] .+ a) : (values..., params[key])
    end
    NamedTuple{_keys}(values)
end

function make_alternative_parameters_by_adding_constant(params::NamedTuple, n::Integer, a_vals::Vector{<:Number})
    [make_alternative_parameters_by_adding_constant(params,n, a) for a ∈ a_vals]
end

alternative_parameters = make_alternative_parameters_by_adding_constant(default_parameters, 1, Vector(.1:.1:10.))

h_ensemble = hodeensemble(; parameters = alternative_parameters)
sol = integrate(h_ensemble, ImplicitMidpoint())
dl = ParametricDataLoader(sol)
batch = Batch(100)
arch = GeneralizedHamiltonianArchitecture(2; parameters = default_parameters)
nn = NeuralNetwork(arch)
o = Optimizer(AdamOptimizer(), nn)
o(nn, dl, batch)