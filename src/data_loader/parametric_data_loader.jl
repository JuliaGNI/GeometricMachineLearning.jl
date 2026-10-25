@doc raw"""
    ParametricDataLoader(data, parameters)
    ParametricDataLoader(ensemble_solution)

Make a loader of trajectories of a system together with the parameters ``\mu`` of the system that
each trajectory belongs to.

`data` is a `2d × nt × m` array, or a `(q, p)` named tuple of two `d × nt × m` arrays of one
array type: `m` trajectories of `nt` time steps, with `q` above `p` where they are stacked.
`parameters` is a vector of `m` named tuples, the ``\mu`` of each trajectory. Every ``\mu`` has the
keys of the first, in the same order and at every level, and the same sizes of its leaves. The data
and every ``\mu`` have one element type. An `EnsembleSolution` of a problem with `(q, p)` states
gives its `q` and `p` and the parameters of its problem.

# The batch

A `Batch(batch_size)` draws, in one epoch, each pair of a start position and a trajectory once, in
minibatches of `batch_size`; the last minibatch is shorter. A drawn minibatch is the triple
`(input, output, μs)`: the state at the start position, the state one time step later, and the
``\mu`` of each sample. [`train!`](@ref) on a `ParametricDataLoader` trains on
[`ParametricLoss`](@ref)`()` when no `loss` is given, so the network is applied with
[`apply_parametric`](@ref), one time step per sample. A `Batch` with a sequence length or a
prediction window other than one is an `ArgumentError`.

A problem without system parameters is trained with a [`DataLoader`](@ref).
"""
struct ParametricDataLoader{
    T, AT <: ArrayOrNamedTuple{T}, PT <: AbstractVector{<:NamedTuple}}
    input::AT
    parameters::PT
    input_time_steps::Int
    n_params::Int

    function ParametricDataLoader(input::AT,
            parameters::PT) where {T,
            AT <: Union{AbstractArray{T, 3},
                NamedTuple{(:q, :p), Tuple{AT3, AT3}} where {AT3 <: AbstractArray{T, 3}}},
            PT <: AbstractVector{<:NamedTuple}}
        _, input_time_steps, n_params = data_size(input)
        n_params ≥ 1 && length(parameters) == n_params || throw(ArgumentError(
            "the data hold $(n_params) trajectories and there are $(length(parameters)) " *
            "system parameters; a `ParametricDataLoader` takes one for each trajectory, " *
            "and at least one trajectory."))
        layout = parameterlayout(first(parameters))
        for (j, μ) in enumerate(parameters)
            S = parameter_eltype(μ)
            S === T || S === Union{} ||
                throw(ArgumentError(
                    "the data have the element type $(T) and a system parameter has the " *
                    "element type $(S); a `ParametricDataLoader` takes one element type."))
            parameterlayout(μ) == layout || throw(ArgumentError(
                "system parameter $(j) differs from system parameter 1 in its keys, their " *
                "order or the sizes of its leaves; a `ParametricDataLoader` takes one shape."))
        end
        new{T, AT, PT}(input, parameters, input_time_steps, n_params)
    end
end

data_size(input::AbstractArray{<:Any, 3}) = size(input)

function data_size(input::NamedTuple{(:q, :p)})
    size(input.q) == size(input.p) || throw(ArgumentError(
        "`q` has the size $(size(input.q)) and `p` has the size $(size(input.p))."))
    size(input.q)
end

function ParametricDataLoader(ensemble_solution::EnsembleSolution{T,
        T1,
        Vector{ST}}) where {
        T, T1, TuT, TT <: TimeSeries{T1},
        ST <: GeometricSolution{T, T1, TT, NamedTuple{(:t, :q, :p, :q̇, :ṗ), TuT}}}
    ParametricDataLoader(
        (q = data_tensor([s.q for s in ensemble_solution.s]),
            p = data_tensor([s.p for s in ensemble_solution.s])),
        ensemble_solution.problem.parameters)
end

function networkbackend(dl::ParametricDataLoader{T, <:AbstractArray}) where {T}
    networkbackend(dl.input)
end
function networkbackend(dl::ParametricDataLoader{T, <:NamedTuple}) where {T}
    networkbackend(dl.input.q)
end

# The start positions are those with a target `prediction_window` steps later; the network takes one
# time step per sample and returns one, so the sequence length and the prediction window are one.
function (batch::Batch)(dl::ParametricDataLoader)
    batch.seq_length == 1 || throw(ArgumentError(
        "a `ParametricDataLoader` draws one time step per sample, and the `Batch` asks for a " *
        "sequence length of $(batch.seq_length)."))
    batch.prediction_window == 1 || throw(ArgumentError(
        "a `ParametricDataLoader` draws a target of one time step per sample, and the `Batch` " *
        "asks for a prediction window of $(batch.prediction_window)."))
    n = dl.input_time_steps - batch.prediction_window
    n ≥ 1 || throw(ArgumentError(
        "the `ParametricDataLoader` has $(dl.input_time_steps) time steps and the `Batch` asks " *
        "for a prediction window of $(batch.prediction_window), which leaves no minibatch."))
    batch_over_two_axes(batch, n, dl.n_params)
end

# The input at the start position, the target `seq_length` steps later, and the `μ` of the trajectory
# of each sample, which row 2 of the batch indices names.
function convert_input_and_batch_indices_to_array(dl::ParametricDataLoader, batch::Batch,
        batch_indices_tuple::Vector{Tuple{Int, Int}})
    indices = convert_vector_of_tuples_to_matrix(networkbackend(dl), batch_indices_tuple)
    (gather(dl.input, indices, 0, batch.seq_length),
        gather(dl.input, indices, batch.seq_length, batch.prediction_window),
        [dl.parameters[trajectory] for (_, trajectory) in batch_indices_tuple])
end
