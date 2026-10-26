# The batching rules of `src/data_loader/batch.jl`. The data are ten samples
# `x = T.(reshape(1:20, 2, 10))` with `y = 3 .* x`, so that `output == 3 .* input` identifies the
# pair of one sample and not the pair of another, and a drawn index pair names the sample it draws.

using GeometricMachineLearning
using GeometricMachineLearning: convert_input_and_batch_indices_to_array, number_of_batches
using Test
import Random

Random.seed!(1234)

# The kind of a loader is its fourth type parameter.
kind(dl::DataLoader) = typeof(dl).parameters[4]

samples(T) = T.(reshape(1:20, 2, 10))

# What one minibatch holds, for every minibatch of an epoch.
gathered(dl, batch) = map(batch(dl)) do indices
    convert_input_and_batch_indices_to_array(dl, batch, indices)
end

@testset "a supervised loader samples every pair, the last included ($T)" for T in (Float32, Float64)
    x = samples(T)
    y = 3 .* x
    loaders = ("input and output" => DataLoader(x, y; suppress_info = true),
        "a view output" => DataLoader(x, view(y, :, :); suppress_info = true))
    for (name, dl) in loaders
        @test kind(dl) === :RegularData
        @test (dl.input_dim, dl.input_time_steps, dl.n_params) == (2, 10, 1)
        @test (dl.output_dim, dl.output_time_steps) == (2, 10)

        batch = Batch(3)
        minibatches = batch(dl)
        # every sample is drawn, once, the tenth included
        @test sort(vcat(minibatches...)) == [(t, 1) for t in 1:10]
        @test length(minibatches) == 4
        @test number_of_batches(dl, batch) == 4

        data = gathered(dl, batch)
        inputs = first.(data)
        @test sort(size.(inputs, 3)) == [1, 3, 3, 3]
        @test all(size(input)[1:2] == (2, 1) for input in inputs)
        for (input, output) in data
            @test eltype(input) === T
            @test eltype(output) === T
            @test size(output) == size(input)
            @test output == 3 .* input
        end
        # the gathered inputs are the columns of `x`, in some order
        @test sort(eachcol(hcat((input[:, 1, k]
        for input in inputs for k in axes(input, 3))...))) ==
              sort(eachcol(Matrix(x)))
    end
end

# The `(q, p)` loader of the same ten samples: `autoencoder = true` reads the second axis of a
# matrix pair as the parameter index, so its one time step carries all ten samples.
@testset "a (q, p) loader of samples draws every pair ($T)" for T in (Float32, Float64)
    x = samples(T)
    dl = DataLoader((q = x, p = 3 .* x); autoencoder = true, suppress_info = true)
    @test kind(dl) === :RegularData
    @test (dl.input_dim, dl.input_time_steps, dl.n_params) == (4, 1, 10)

    batch = Batch(3)
    drawn = batch(dl)
    @test sort(vcat(drawn...)) == [(1, p) for p in 1:10]
    minibatches = map(
        indices -> convert_input_and_batch_indices_to_array(dl, batch, indices),
        drawn)
    @test length(minibatches) == 4
    @test sort([size(mb.q, 3) for mb in minibatches]) == [1, 3, 3, 3]
    for (indices, mb) in zip(drawn, minibatches)
        @test mb isa NamedTuple{(:q, :p)}
        @test eltype(mb.q) === T
        @test eltype(mb.p) === T
        @test mb.p == 3 .* mb.q
        # each slot holds the sample its own drawn index names, so a `q` and a `p` read from the
        # same other column shows here
        @test all(mb.q[:, 1, k] == x[:, indices[k][2]] for k in axes(mb.q, 3))
    end
    # the gathered columns are the columns of `x`, in some order
    @test sort(eachcol(hcat((mb.q[:, 1, k]
    for mb in minibatches for k in axes(mb.q, 3))...))) ==
          sort(eachcol(Matrix(x)))
end

# One sample is a loader the training loop can draw from.
@testset "one sample is one minibatch ($T)" for T in (Float32, Float64)
    x = T.(reshape(1:2, 2, 1))
    dl = DataLoader(x, x; suppress_info = true)
    batch = Batch(1)
    minibatches = batch(dl)
    @test length(minibatches) == 1
    @test only(minibatches) == [(1, 1)]
    input, output = only(gathered(dl, batch))
    @test size(input) == (2, 1, 1)
    @test eltype(input) === T
    @test eltype(output) === T
    @test output == input
end

# One count, `cld(n · n_params, batch_size)` with `n` the number of start positions, and an
# `ArgumentError` where no window fits or the parameter axis is empty.
@testset "one count, and a window that fits no start position" begin
    trajectory = DataLoader(rand(Float64, 2, 10); suppress_info = true)
    @test kind(trajectory) === :TimeSeries
    @test length(Batch(3, 9, 1)(trajectory)) == 1
    @test_throws ArgumentError Batch(3, 10, 1)(trajectory)

    x = samples(Float64)
    dl = DataLoader(x, 3 .* x; suppress_info = true)
    @test length(Batch(3, 10, 10)(dl)) == 1
    @test_throws ArgumentError Batch(3, 11, 1)(dl)

    # the message names the time steps, the sequence length and the prediction window: three
    # different numbers here, so that each one's presence is a separate reading
    raised = try
        Batch(3, 11, 5)(trajectory)
    catch e
        e
    end
    @test raised isa ArgumentError
    for word in ("time steps", "sequence length", "prediction window")
        @test occursin(word, raised.msg)
    end
    @test occursin("$(trajectory.input_time_steps)", raised.msg)
    @test occursin("11", raised.msg)
    @test occursin("5", raised.msg)

    # a loader whose parameter axis is empty draws no minibatch either
    @test_throws ArgumentError Batch(1)(DataLoader(zeros(Float64, 2, 0);
        autoencoder = true, suppress_info = true))
    @test_throws ArgumentError Batch(1)(DataLoader(zeros(Float64, 2, 3, 0); suppress_info = true))

    # where the output's time steps bind the window, the message names them
    with_output = DataLoader(rand(Float64, 2, 10, 3), rand(Float64, 2, 1, 3);
        suppress_info = true)
    raised = try
        Batch(2, 1, 2)(with_output)
    catch e
        e
    end
    @test raised isa ArgumentError
    @test occursin("10 time steps and 1 in its output", raised.msg)
end

# The classifier's loader, and `Batch(batch_size, dl)` — one draw per sample with the whole
# sequence and the whole output, which is what GMLDatasets calls.
@testset "Batch(batch_size, dl) draws every sample with the whole sequence ($T)" for T in (Float32, Float64)
    n_classes, n_patches, n_images = 10, 4, 10
    input = rand(T, 9, n_patches, n_images)
    output = zeros(Int, n_classes, 1, n_images)
    for (k, c) in pairs(rand(1:n_classes, n_images))
        output[c, 1, k] = 1
    end
    dl = DataLoader(input, output; suppress_info = true)
    @test kind(dl) === :RegularData
    @test (dl.input_dim, dl.input_time_steps, dl.n_params) == (9, 4, 10)
    @test (dl.output_dim, dl.output_time_steps) == (10, 1)

    batch = Batch(2, dl)
    @test batch == Batch(2, 4, 1)
    minibatches = batch(dl)
    @test length(minibatches) == 5
    # `Batch(2)` would pair the first patch with the one output column, so it raises
    @test_throws ArgumentError Batch(2)(dl)
    @test_throws "Batch(batch_size, dl)" number_of_batches(dl, Batch(2))
    for indices in minibatches
        input_batch, output_batch = convert_input_and_batch_indices_to_array(dl, batch, indices)
        @test size(input_batch) == (9, 4, 2)
        @test size(output_batch) == (10, 1, 2)
        @test eltype(input_batch) === T
        @test eltype(output_batch) === Int
        @test output_batch == output[:, :, [i[2] for i in indices]]
    end
end

# The target is the output when there is one, whatever kind the loader is typed as. A
# `DataLoader{T, A3, Array{Int, 3}, :TimeSeries}` is what GMLDatasets and two tests build by naming
# the type, and it must batch like the `:RegularData` loader of the same data.
@testset "a loader with an output is samples whatever its kind ($T)" for T in (Float32, Float64)
    input = rand(T, 9, 4, 10)
    output = zeros(Int, 10, 1, 10)
    for (k, c) in pairs(rand(1:10, 10))
        output[c, 1, k] = 1
    end
    dl = DataLoader(input, output; suppress_info = true)
    dl_time_series = DataLoader{T, typeof(input), typeof(output), :TimeSeries}(
        input, output, 9, 4, 10, 10, 1)

    for batch in (Batch(2, dl), Batch(3, 1, 1))
        Random.seed!(11)
        asked = batch(dl_time_series)
        Random.seed!(11)
        reference = batch(dl)
        @test asked == reference
        for (i, j) in zip(asked, reference)
            here = convert_input_and_batch_indices_to_array(dl_time_series, batch, i)
            @test here == convert_input_and_batch_indices_to_array(dl, batch, j)
            @test eltype(here[1]) === T
            @test eltype(here[2]) === Int
        end
    end
end

# The `Batch` method of `integrate` exists to carry one number: it is the three-argument form a
# training script writes, and it has to give the trajectory the keyword form gives. The window is
# chosen so that the two disagree when it is dropped: a standard transformer of system dimension
# 5 on a four-column input takes two predictions of two columns for `n_points = 8`, where the
# default window takes one of four columns. Both forms carry the input's element type, so a
# trajectory copied to the host or plotted keeps its precision.
@testset "the Batch method of integrate forwards its prediction window ($T)" for T in (Float32, Float64)
    nn = NeuralNetwork(StandardTransformerIntegrator(5), CPU(), T)
    ics = rand(T, 5, 4)

    @test Batch(3, 5, 2) == Batch{:Transformer}(3, 5, 2)
    @test eltype(integrate(nn, ics, Batch(3, 5, 2); n_points = 5)) === T
    @test eltype(integrate(nn, ics; n_points = 5, prediction_window = 2)) === T
    @test integrate(nn, ics, Batch(3, 5, 2); n_points = 5) ==
          integrate(nn, ics; n_points = 5, prediction_window = 2)
    @test integrate(nn, ics, Batch(3, 5, 2); n_points = 8) ==
          integrate(nn, ics; n_points = 8, prediction_window = 2)
    @test integrate(nn, ics, Batch(3, 5, 2); n_points = 8) ≠
          integrate(nn, ics; n_points = 8)
end
