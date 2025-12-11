"""
    SymbolicPullback(arch::HamiltonianArchitecture)

Make a `SymbolicPullback` based on a [`HamiltonianArchitecture`](@ref).

# Implementation

Internally this is calling `SymbolicNeuralNetwork` and [`HNNLoss`](@ref).
"""
function SymbolicPullback(arch::HamiltonianArchitecture)
    nn = SymbolicNeuralNetwork(arch)
    loss = HNNLoss(arch)
    SymbolicPullback(nn, loss)
end

# TODO: This constitutes type piracy - should be in `SymbolicNeuralNetworks`!
# TODO: note that the the whole concatenation business may not be the ideal solution, but implementing a separate build_nn_function for any number of input arguments isn't either. Find something better here!!
function (_pullback::SymbolicPullback)(ps, model, input_and_parameters::Tuple{<:QPTOAT, <:QPTOAT, <:Union{AbstractVector, NamedTuple}})::Tuple
    input, output, system_params = input_and_parameters
    _pullback.loss(model, ps, concatenate_array_with_parameters(input, system_params), output), _pullback.fun(input, output, system_params, ps)
end

function assign_q_and_p(z::SymbolicNeuralNetworks.Symbolics.Arr{SymbolicNeuralNetworks.Symbolics.Num, 1}, N::Int)
    @assert length(z) == 2N
    (q = z[1:N], p = z[(N+1):2N])
end

function SymbolicNeuralNetworks.Symbolics.SymbolicUtils.Code.create_array(::Type{<:SymbolicNeuralNetworks.Symbolics.Arr}, ::Nothing, ::Val, ::Val{dims}, elems...) where {dims}
    SymbolicNeuralNetworks.Symbolics.Arr([elems...])
end

function ParameterHandling.flatten(::Type{T}, ps::NeuralNetworkParameters) where {T<:Real}
    _ps = NamedTuple{keys(ps)}(values(ps))
    x_vec, unflatten_to_NamedTuple = ParameterHandling.flatten(T, _ps)
    function unflatten_to_NeuralNetworkParameters(v::Vector{T})
        nt = unflatten_to_NamedTuple(v)
        NeuralNetworkParameters{keys(nt)}(values(nt))
    end
    x_vec, unflatten_to_NeuralNetworkParameters
end
function ParameterHandling.flatten(T::Type{SymbolicNeuralNetworks.Symbolics.Num}, v::SymbolicNeuralNetworks.Symbolics.Arr{SymbolicNeuralNetworks.Symbolics.Num, 1})
    ParameterHandling.flatten(T, [elem for elem ∈ v])
end

"""
    semi_flatten_network_parameters(params)

Should be used together with [`GeneralizedHamiltonianArchitecture`](@ref) and `SymbolicPullback`s.
"""
function semi_flatten_network_parameters(::Type{T}, params::NeuralNetworkParameters) where {T}
    _values = Tuple(ParameterHandling.flatten(T, value)[1] for value ∈ values(params))
    NeuralNetworkParameters{keys(params)}(_values)
end

"""
    SymbolicPullback(nn, loss, system_params)

The `SymbolicPullback` for the case when we have a parametrized system (with `system_params`).
"""
function SymbolicPullback(nn::NeuralNetwork, loss::ParametricLoss, system_params::OptionalParameters)
    cache = Dict()
    symbolic_system_parameters = SymbolicNeuralNetworks.symbolize!(cache, system_params, :S)
    cache = Dict()
    symbolic_network_parameters = SymbolicNeuralNetworks.symbolize!(cache, nn.params, :W)

    input_dim = input_dimension(nn.model)
    output_dim = output_dimension(nn.model)
    flattened_params = ParameterHandling.flatten(SymbolicNeuralNetworks.Symbolics.Num, symbolic_system_parameters)
    SymbolicNeuralNetworks.Symbolics.@variables sinput[1:(input_dim + length(system_params))]
    SymbolicNeuralNetworks.Symbolics.@variables soutput[1:output_dim]
    symbolic_loss = loss(nn.model, symbolic_network_parameters, sinput[1:input_dim], soutput, flattened_params[2]([elem for elem ∈ sinput[(input_dim+1):end]]))
    symbolic_diffs = SymbolicNeuralNetworks.symbolic_differentials(symbolic_network_parameters)
    symbolic_pullbacks = [SymbolicNeuralNetworks.symbolic_derivative(f_single, symbolic_diffs) for f_single ∈ symbolic_loss]

    semi_flattened_network_parameters = semi_flatten_network_parameters(SymbolicNeuralNetworks.Symbolics.Num, symbolic_network_parameters)
    pbs_executable = SymbolicNeuralNetworks.build_nn_function(symbolic_pullbacks, semi_flattened_network_parameters, sinput, soutput)
    function pbs(input, output, params)
        T = (typeof(input) <: QPT) ? eltype(input.q) : eltype(input)
        pullback(::Union{Real, AbstractArray{<:Real}}) = _get_contents(_get_params(pbs_executable(input, output, semi_flatten_network_parameters(T, params))))
        pullback
    end
    SymbolicPullback(loss, pbs)
end

function (_pullback::SymbolicPullback)(ps, model, input_and_parameters::Tuple{<:AbstractArray, <:AbstractArray, <:Union{NamedTuple, AbstractVector}})::Tuple
    input, output, system_params = input_and_parameters
    _pullback(ps, model, (concatenate_array_with_parameters(input, system_params), output))
end

function (_pullback::SymbolicPullback)(ps, model, input_and_parameters::Tuple{AT, AT, <:Union{NamedTuple, AbstractVector}})::Tuple where {T, AT <: AbstractArray{T, 3}}
    input, output, system_params = input_and_parameters
    _input = reshape(input, size(input, 1), size(input, 2) * size(input, 3))
    _output = reshape(output, size(output, 1), size(output, 2) * size(output, 3))
    _pullback.loss(model, ps, _input, _output, system_params), _pullback.fun(concatenate_array_with_parameters(_input, system_params), _output, ps)
end

function (_pullback::SymbolicPullback)(ps, model, input_and_parameters::Tuple{<:QPT, <:QPT, <:Union{NamedTuple, AbstractVector}})::Tuple
    input, output, system_params = input_and_parameters
    _pullback(ps, model, (vcat(input.q, input.p), vcat(output.q, output.p), system_params))
end