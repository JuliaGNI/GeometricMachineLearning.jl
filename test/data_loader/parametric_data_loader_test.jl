using GeometricMachineLearning
using GeometricMachineLearning: convert_input_and_batch_indices_to_array
using Test
using GeometricProblems.CoupledHarmonicOscillator: hodeensemble, default_parameters
using GeometricIntegrators: ImplicitMidpoint, integrate
using Random: seed!
seed!(123)

function make_alternative_parameters_by_adding_constant(params::NamedTuple = default_parameters(),
        a::Number = 1.)
    NamedTuple{keys(params)}(Tuple(value .+ a for value in values(params)))
end

all_parameters = [default_parameters(), make_alternative_parameters_by_adding_constant()]

h_ensemble = hodeensemble(; parameters = all_parameters)
sol = integrate(h_ensemble, ImplicitMidpoint())
dl = ParametricDataLoader(sol)
batch = Batch(2)
batch_indices = batch(dl)

# Each entry of a batch is a `(time index, parameter index)` pair, and the third element of what
# `convert_input_and_batch_indices_to_array` returns has to be the parameters of *that* trajectory.
# The batches are shuffled, so this asserts the correspondence rather than which batch holds which
# parameters -- pinning the latter makes the test depend on the RNG stream of the Julia version.
function batch_is_consistent(n::Integer)
    input, output, parameters = convert_input_and_batch_indices_to_array(dl, batch, batch_indices[n])
    all(enumerate(batch_indices[n])) do (k, (time_index, parameter_index))
        parameters[k] == all_parameters[parameter_index] &&
            input.q[:, 1, k] == dl.input.q[:, time_index, parameter_index] &&
            input.p[:, 1, k] == dl.input.p[:, time_index, parameter_index] &&
            output.q[:, 1, k] == dl.input.q[:, time_index + 1, parameter_index] &&
            output.p[:, 1, k] == dl.input.p[:, time_index + 1, parameter_index]
    end
end

@test all(batch_is_consistent, eachindex(batch_indices))

# Both parameter sets have to turn up somewhere, or the assertion above would also pass on a data
# loader that always returned the first one.
returned_parameters = Set(parameters
                          for n in eachindex(batch_indices)
                          for parameters in last(convert_input_and_batch_indices_to_array(
                              dl, batch, batch_indices[n])))
@test returned_parameters == Set(all_parameters)
