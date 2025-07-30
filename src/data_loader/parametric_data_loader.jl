"""
    ParametricDataLoader

Very similar to [`DataLoader`](@ref), but can deal with parametric problems.
"""
struct ParametricDataLoader{T, AT<:QPTOAT2, VT<:AbstractVector}
    input::AT
    input_dim::Int
    input_time_steps::Int
    parameters::VT
    n_params::Int

    function ParametricDataLoader(data::QPTOAT2{T, 3}, parameters::AbstractVector) where {T}
        input_dim, input_time_steps, n_params = _size(data)
        @assert T == _eltype(parameters) "Provided data and parameters must have the same eltype!"
        @assert length(parameters) == _size(data, 3) "The number of provided parameters and the parameter axis of the supplied data do not have the same length!"

        new{T, typeof(data), typeof(parameters)}(data, input_dim, input_time_steps, parameters, n_params)
    end
end

function ParametricDataLoader(input::AbstractMatrix{T}, parameters::AbstractVector) where {T}
    ParametricDataLoader(reshape(input, size(input)..., 1), parameters)
end

function ParametricDataLoader(ensemble_solution::EnsembleSolution{T, T1, Vector{ST}}) where {T, 
                             T1, 
                             DT <: DataSeries{T}, 
                             ST <: GeometricSolution{T, T1, NamedTuple{(:q, :p), Tuple{DT, DT}}}
                             }

    sys_dim = length(ensemble_solution.s[1].q[0])
    input_time_steps = length(ensemble_solution.t)
    n_params = length(ensemble_solution.s)
    params = ensemble_solution.problem.parameters

    data = (q = zeros(T, sys_dim, input_time_steps, n_params), p = zeros(T, sys_dim, input_time_steps, n_params))

    for (solution, i) in zip(ensemble_solution.s, axes(ensemble_solution.s, 1))
        for dim in 1:sys_dim 
            data.q[dim, :, i] = solution.q[:, dim]
            data.p[dim, :, i] = solution.p[:, dim]
        end 
    end

    ParametricDataLoader(data, params)
end

# """
#     rearrange_parameters(parameters)
# 
# Rearrange `parameters` such that they can be used by [`ParametricDataLoader`](@ref).
# """
# function rearrange_parameters(parameters::Vector{<:NamedTuple})
#     parameters_rearranged = zeros(_eltype(parameters), )
# end

# function batch_over_two_axes(batch::Batch, number_columns::Int, third_dim::Int, dl::ParametricDataLoader)
#     time_indices = shuffle(1:number_columns)
#     parameter_indices = shuffle(1:third_dim)
#     complete_indices = Iterators.product(time_indices, parameter_indices) |> collect |> vec
#     batches = ()
#     n_batches = number_of_batches(dl, batch)
#     for batch_number in 1:(n_batches - 1)
#         batches = (batches..., complete_indices[(batch_number - 1) * batch.batch_size + 1 : batch_number * batch.batch_size])
#     end
#     (batches..., complete_indices[(n_batches - 1) * batch.batch_size + 1:end])
# end

function optimize_for_one_epoch!(   opt::Optimizer, 
                                    model, 
                                    ps::Union{NeuralNetworkParameters, NamedTuple}, 
                                    dl::ParametricDataLoader{T}, 
                                    batch::Batch, 
                                    _pullback::AbstractPullback, 
                                    λY) where T
    count = 0
    total_error = T(0)
    batches = batch(dl)
    for batch_indices in batches 
        count += 1
        # these `copy`s should not be necessary! coming from a Zygote problem!
        _input_nt_output_nt_parameter_indices = convert_input_and_batch_indices_to_array(dl, batch, batch_indices)
        # input_nt_output_nt = _input_nt_output_nt_parameter_indices[1:2]
        loss_value, pullback = _pullback(ps, model, _input_nt_output_nt_parameter_indices)
        total_error += loss_value
        dp = _get_params(_unpack_tuple(pullback(one(loss_value))))
        optimization_step!(opt, λY, ps, dp)
    end
    total_error / count
end

function parameter_indices(parameters::AbstractVector, parameter_indices::AbstractVector{Int})
    [parameters[parameter_index] for parameter_index in parameter_indices]
end

function parameter_indices(parameters::AbstractVector, batch_indices::AbstractMatrix{Int})
    parameter_indices(parameters, batch_indices[2, :])
end

function parameter_indices(dl::ParametricDataLoader, indices::AbstractArray{Int})
    parameter_indices(dl.parameters, indices)
end

function convert_input_and_batch_indices_to_array(dl::ParametricDataLoader{T, BT}, batch::Batch, batch_indices_tuple::Vector{Tuple{Int, Int}}) where {T, AT<:AbstractArray{T, 3}, BT<:NamedTuple{(:q, :p), Tuple{AT, AT}}}
    backend = networkbackend(dl.input.q)
    
    # the batch size is smaller for the last batch
    _batch_size = length(batch_indices_tuple)

    batch_indices = convert_vector_of_tuples_to_matrix(backend, batch_indices_tuple)

    q_input = KernelAbstractions.allocate(backend, T, dl.input_dim ÷ 2, batch.seq_length, _batch_size)
    p_input = similar(q_input)

    assign_input_from_vector_of_tuples! = assign_input_from_vector_of_tuples_kernel!(backend)
    assign_input_from_vector_of_tuples!(q_input, p_input, dl.input, batch_indices, ndrange=(dl.input_dim ÷ 2, batch.seq_length, _batch_size))

    q_output = KernelAbstractions.allocate(backend, T, dl.input_dim ÷ 2, batch.prediction_window, _batch_size)
    p_output = similar(q_output)

    assign_output_from_vector_of_tuples! = assign_output_from_vector_of_tuples_kernel!(backend)
    assign_output_from_vector_of_tuples!(q_output, p_output, dl.input, batch_indices, batch.seq_length, ndrange=(dl.input_dim ÷ 2, batch.prediction_window, _batch_size))

    (q = q_input, p = p_input), (q = q_output, p = p_output), parameter_indices(dl, batch_indices)
end

function number_of_batches(dl::ParametricDataLoader, batch::Batch)
    @assert dl.input_time_steps ≥ (batch.seq_length + batch.prediction_window) "The number of time steps has to be greater than sequence length + prediction window."
    Int(ceil((dl.input_time_steps - (batch.seq_length - 1) - batch.prediction_window) * dl.n_params / batch.batch_size))
end

function (batch::Batch)(dl::ParametricDataLoader)
    batch_over_two_axes(batch, dl.input_time_steps - (batch.seq_length - 1) - batch.prediction_window, dl.n_params, number_of_batches(dl, batch))
end

function (o::Optimizer)(nn::NeuralNetwork{<:GeneralizedHamiltonianArchitecture}, dl::ParametricDataLoader, batch::Batch{:FeedForward}, n_epochs::Int=1; kwargs...)
    loss = ParametricLoss()
    o(nn, dl, batch, n_epochs, loss; kwargs...)
end