using GeometricMachineLearning
using GeometricMachineLearning: ParametricDataLoader, ParametricLoss, params,
                                convert_input_and_batch_indices_to_array
using GeometricProblems.CoupledHarmonicOscillator: hodeensemble, default_parameters
using GeometricIntegrators: ImplicitMidpoint, integrate
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/parametric_affine.jl")

# `m` trajectories of `nt` time steps of a state of dimension `2d`.
const d, nt, m = 1, 5, 3

# The data of trajectory `j` at time step `t`, in every entry, and its system parameters `μⱼ`.
constant_data(::Type{T}) where {T} = [T(j) for _ in 1:(2d), t in 1:nt, j in 1:m]
timed_data(::Type{T}) where {T} = [T(100j + t) for _ in 1:(2d), t in 1:nt, j in 1:m]
system_parameters(::Type{T}) where {T} = [(a = T(j),) for j in 1:m]

# The loader of a `2d × nt × m` array, and of the `(q, p)` pair of its two halves.
function loaders(data, μs)
    (ParametricDataLoader(data, μs),
        ParametricDataLoader((q = data[1:d, :, :], p = data[(d + 1):end, :, :]), μs))
end

# Entry `k` of a drawn batch, as the vector of the state it holds.
sample(x::AbstractArray, k) = x[:, 1, k]
sample(x::NamedTuple, k) = vcat(x.q[:, 1, k], x.p[:, 1, k])

# Every sample of one epoch: its start position and trajectory, its input, its output and its `μ`.
function epoch(dl, batch)
    samples = []
    for indices in batch(dl)
        input, output, μs = convert_input_and_batch_indices_to_array(dl, batch, indices)
        @test length(μs) == length(indices)
        for (k, index) in enumerate(indices)
            push!(samples, (index, sample(input, k), sample(output, k), μs[k]))
        end
    end
    samples
end

all_pairs = sort(vec([(t, j) for t in 1:(nt - 1), j in 1:m]))

@testset "one epoch of Batch(b) on a ParametricDataLoader ($T)" for T in REAL_ELTYPES
    Random.seed!(1)
    for dl in loaders(constant_data(T), system_parameters(T))
        samples = epoch(dl, Batch(4))
        # each pair of a start position and a trajectory once
        @test sort([index for (index, _, _, _) in samples]) == all_pairs
        for (_, input, output, μ) in samples
            @test μ.a == input[1]
            @test all(==(input[1]), input)
        end
        # the last minibatch is shorter: 12 pairs in minibatches of 5
        @test length.(Batch(5)(dl)) == [5, 5, 2]
    end
    for dl in loaders(timed_data(T), system_parameters(T))
        samples = epoch(dl, Batch(4))
        @test sort([index for (index, _, _, _) in samples]) == all_pairs
        for ((t, j), input, output, μ) in samples
            @test input == fill(T(100j + t), 2d)
            # the output is the input one time step later
            @test output == input .+ 1
            @test μ.a == T(j)
            @test eltype(input) === eltype(output) === T
        end
    end
end

@testset "a ParametricDataLoader of an EnsembleSolution" begin
    parameters = [default_parameters(), map(v -> v .+ 1, default_parameters())]
    solution = integrate(hodeensemble(; parameters = parameters), ImplicitMidpoint())
    dl = ParametricDataLoader(solution)
    @test dl.parameters == parameters
    @test dl.n_params == 2
    n_steps = length(solution.t)
    @test dl.input_time_steps == n_steps
    for j in 1:2, t in 1:n_steps

        @test dl.input.q[:, t, j] == solution.s[j].q[t - 1]
        @test dl.input.p[:, t, j] == solution.s[j].p[t - 1]
    end
    Random.seed!(2)
    samples = epoch(dl, Batch(7))
    @test sort([index for (index, _, _, _) in samples]) ==
          sort(vec([(t, j) for t in 1:(n_steps - 1), j in 1:2]))
    for ((t, j), input, output, μ) in samples
        @test μ == parameters[j]
        @test input == vcat(dl.input.q[:, t, j], dl.input.p[:, t, j])
        @test output == vcat(dl.input.q[:, t + 1, j], dl.input.p[:, t + 1, j])
    end
end

# The message of the `ArgumentError` that `f` throws.
function argument_error_message(f)
    try
        f()
    catch error
        error isa ArgumentError && return error.msg
        rethrow()
    end
    "no error"
end

@testset "the edges of a ParametricDataLoader" begin
    data = timed_data(Float64)
    μs = system_parameters(Float64)
    dl = ParametricDataLoader(data, μs)

    # one time step per sample: a sequence length other than one is an error before any step
    message = argument_error_message(() -> Batch(2, 3, 1)(dl))
    @test occursin("sequence length of 3", message)
    # a prediction window other than one is an error before any step, with the time steps to spare
    message = argument_error_message(() -> Batch{:FeedForward}(2, 1, 2)(dl))
    @test occursin("prediction window of 2", message)

    # a parameter vector of the wrong length, and no trajectory
    message = argument_error_message(() -> ParametricDataLoader(data, μs[1:2]))
    @test occursin("hold 3 trajectories", message) &&
          occursin("there are 2 system parameters", message)
    message = argument_error_message(
        () -> ParametricDataLoader(zeros(2d, nt, 0), NamedTuple{(:a,), Tuple{Float64}}[]))
    @test occursin("hold 0 trajectories", message)

    # too few time steps for the batch
    message = argument_error_message(() -> Batch(2)(ParametricDataLoader(data[:, 1:1, :], μs)))
    @test occursin("1 time step", message)

    # one element type for the data and every `μ`
    message = argument_error_message(() -> ParametricDataLoader(Float32.(data), μs))
    @test occursin("Float32", message) && occursin("Float64", message)
    @test ParametricDataLoader(data, [(a = 1.0, b = [2.0, 3.0]) for _ in 1:m]).n_params == m
    # an empty `μ` has no element type, and concatenates nothing
    @test ParametricDataLoader(data, [(;) for _ in 1:m]).n_params == m

    # `q` and `p` of one size
    message = argument_error_message(
        () -> ParametricDataLoader((q = data[1:1, :, :], p = data[:, :, :]), μs))
    @test occursin("(1, $(nt), $(m))", message) && occursin("(2, $(nt), $(m))", message)

    # `q` and `p` of one array type, as `gather` batches them
    @test_throws MethodError ParametricDataLoader(
        (q = data[1:d, :, :], p = view(data, (d + 1):(2d), :, :)), μs)
    views = ParametricDataLoader(
        (q = view(data, 1:d, :, :), p = view(data, (d + 1):(2d), :, :)), μs)
    input, output, _ = convert_input_and_batch_indices_to_array(views, Batch(2), [(1, 2)])
    @test sample(input, 1) == data[:, 1, 2] && sample(output, 1) == data[:, 2, 2]

    # every `μ` with the keys of the first, in its order and at every level, and its leaf sizes
    for parameters in ([(a = 1.0,), (b = 2.0,), (a = 3.0,)],
        [(a = 1.0, b = 2.0), (b = 3.0, a = 4.0), (a = 5.0, b = 6.0)],
        [(a = 1.0,), (a = 2.0, b = 3.0), (a = 3.0,)],
        [(a = (c = 1.0,),), (a = (e = 2.0,),), (a = (c = 3.0,),)],
        [(a = [1.0, 2.0],), (a = [1.0, 2.0, 3.0],), (a = [1.0, 2.0],)])
        message = argument_error_message(() -> ParametricDataLoader(data, parameters))
        @test occursin("system parameter 2", message)
    end
end

# The network is `(x, μ) ↦ W [x; flatten(μ)] + b`. `train!` without `loss` and with
# `loss = ParametricLoss()` draw the same network and the same minibatches under one seed.
@testset "train! on a ParametricDataLoader trains on ParametricLoss ($T)" for T in REAL_ELTYPES
    dl = ParametricDataLoader(timed_data(T) ./ 100, system_parameters(T))
    function trained(; kwargs...)
        Random.seed!(3)
        nn = NeuralNetwork(Chain(ParametricAffine{2d}(1)), T)
        initial = deepcopy(params(nn))
        opt = TrainingOptimizer(nn.params; algorithm = Adam())
        history = train!(nn, opt, dl, Batch(4), 3; show_progress = false, kwargs...)
        history, initial, params(nn)
    end
    history, initial, final = trained()
    @test history == first(trained(; loss = ParametricLoss()))
    @test length(history) == 3
    @test eltype(history) === T
    @test final.L1.W != initial.L1.W
    @test final.L1.b != initial.L1.b
end
