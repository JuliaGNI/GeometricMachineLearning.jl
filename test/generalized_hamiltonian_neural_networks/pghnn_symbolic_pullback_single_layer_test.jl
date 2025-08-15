using GeometricMachineLearning
using GeometricMachineLearning: Activation, SymplecticEulerB, SymplecticPotentialEnergy, SymbolicNeuralNetwork, ParametricLoss, QPT

params, dim, width, nhidden, activation = (m = 1., ω = π / 2), 2, 2, 1, tanh

se = GeometricMachineLearning.SymbolicPotentialEnergy(dim, width, nhidden, activation; parameters = params)
l = SymplecticEulerB(se; return_parameters=false)
c = Chain(l)
nn = NeuralNetwork(c)

cache = Dict()
symbolic_network_parameters = GeometricMachineLearning.SymbolicNeuralNetworks.symbolize!(cache, nn.params, :W)
cache2 = Dict()
symbolic_system_parameters = GeometricMachineLearning.SymbolicNeuralNetworks.symbolize!(cache2, params, :S)
loss = ParametricLoss()

function GeometricMachineLearning.ParameterHandling.flatten(::Type{T}, ps::NeuralNetworkParameters) where {T<:Real}
    _ps = NamedTuple{keys(ps)}(values(ps))
    x_vec, unflatten_to_NamedTuple = GeometricMachineLearning.ParameterHandling.flatten(T, _ps)
    function unflatten_to_NeuralNetworkParameters(v::Vector{T})
        nt = unflatten_to_NamedTuple(v)
        NeuralNetworkParameters{keys(nt)}(values(nt))
    end
    x_vec, unflatten_to_NeuralNetworkParameters
end
function GeometricMachineLearning.ParameterHandling.flatten(T::Type{GeometricMachineLearning.SymbolicNeuralNetworks.Symbolics.Num}, v::GeometricMachineLearning.SymbolicNeuralNetworks.Symbolics.Arr{GeometricMachineLearning.SymbolicNeuralNetworks.Symbolics.Num, 1})
    GeometricMachineLearning.ParameterHandling.flatten(T, [elem for elem ∈ v])
end

flattened_params = GeometricMachineLearning.ParameterHandling.flatten(GeometricMachineLearning.SymbolicNeuralNetworks.Symbolics.Num, symbolic_system_parameters)
GeometricMachineLearning.SymbolicNeuralNetworks.Symbolics.@variables sinput[1:(dim + length(params))]
GeometricMachineLearning.SymbolicNeuralNetworks.Symbolics.@variables soutput[1:GeometricMachineLearning.output_dimension(nn.model)]
# gen_fun = GeometricMachineLearning.SymbolicNeuralNetworks._build_nn_function(symbolic_pullbacks[1].L1.L1.W, snn.params, GeometricMachineLearning.concatenate_array_with_parameters(snn.input, symbolic_system_parameters), soutput)
symbolic_loss = loss(nn.model, symbolic_network_parameters, sinput[1:dim], soutput, flattened_params[2]([elem for elem ∈ sinput[(dim+1):end]]))
symbolic_diffs = GeometricMachineLearning.SymbolicNeuralNetworks.symbolic_differentials(symbolic_network_parameters)
symbolic_pullbacks = [GeometricMachineLearning.SymbolicNeuralNetworks.symbolic_derivative(f_single, symbolic_diffs) for f_single ∈ symbolic_loss]

"""
    semi_flatten_network_parameters(params)

Should be used together with [`GeneralizedHamiltonianArchitecture`](@ref) and `SymbolicPullback`s.
"""
function semi_flatten_network_parameters(::Type{T}, params::NeuralNetworkParameters) where {T}
    _values = Tuple(GeometricMachineLearning.ParameterHandling.flatten(T, value)[1] for value ∈ values(params))
    NeuralNetworkParameters{keys(params)}(_values)
end

semi_flattened_network_parameters = semi_flatten_network_parameters(GeometricMachineLearning.SymbolicNeuralNetworks.Symbolics.Num, symbolic_network_parameters)
pbs_executable = GeometricMachineLearning.SymbolicNeuralNetworks.build_nn_function(symbolic_pullbacks, semi_flattened_network_parameters, sinput, soutput)
function pbs(input, output, params)
    T = eltype(input)
    pullback(::Union{Real, AbstractArray{<:Real}}) = GeometricMachineLearning._get_contents(GeometricMachineLearning._get_params(pbs_executable(input, output, semi_flatten_network_parameters(T, params))))
    pullback
end

function (_pullback::SymbolicPullback)(ps, model, input_and_parameters::Tuple{<:AbstractArray, <:AbstractArray, <:Union{NamedTuple, AbstractVector}})::Tuple
    input, output, system_params = input_and_parameters
    _pullback(ps, model, (concatenate_array_with_parameters(input, system_params), output))
end
function (_pullback::SymbolicPullback)(ps, model, input_and_parameters::Tuple{<:QPT, <:QPT, <:Union{NamedTuple, AbstractVector}})::Tuple
    input, output, system_params = input_and_parameters
    _pullback(ps, model, (vcat(input.q, input.p), vcat(output.q, output.p), system_params))
end

_pullback = GeometricMachineLearning.SymbolicNeuralNetworks.SymbolicPullback(loss, pbs)
_pullback.fun(rand(4, 10), rand(2, 10), nn.params)(1.)