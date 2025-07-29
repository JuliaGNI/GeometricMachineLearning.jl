using GeometricMachineLearning
using GeometricMachineLearning: convert_input_and_batch_indices_to_array
using Test
using GeometricProblems.CoupledHarmonicOscillator: hodeensemble, default_parameters
using GeometricIntegrators: ImplicitMidpoint, integrate
using Random: seed!
seed!(123)

function make_alternative_parameters_by_adding_constant(params::NamedTuple=default_parameters, a::Number=1.)
    _keys = keys(params)
    values = ()
    for key in _keys
        values = (values..., params[key] .+ a)
    end
    NamedTuple{_keys}(values)
end

alternative_parameters = make_alternative_parameters_by_adding_constant()

h_ensemble = hodeensemble(; parameters = [default_parameters, alternative_parameters])
sol = integrate(h_ensemble, ImplicitMidpoint())
dl = ParametricDataLoader(sol)
batch = Batch(2)
batch_indices = batch(dl)

function test_correct_parameter_splitting(n::Integer, parameters₁ = default_parameters, parameters₂ = alternative_parameters)
    data_in_batch = convert_input_and_batch_indices_to_array(dl, batch, batch_indices[n])
    # the third output and the second datum in the batch
    @test data_in_batch[3][1] == parameters₁
    @test data_in_batch[3][1] != parameters₂
    nothing 
end

test_correct_parameter_splitting(250, default_parameters, alternative_parameters)
test_correct_parameter_splitting(101, alternative_parameters, default_parameters)