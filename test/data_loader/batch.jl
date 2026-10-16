# The batching rules of `src/data_loader/batch.jl`. The data are the part's own: ten samples
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
    @test sort(vcat(batch(dl)...)) == [(1, p) for p in 1:10]
    minibatches = gathered(dl, batch)
    @test length(minibatches) == 4
    @test sort([size(mb.q, 3) for mb in minibatches]) == [1, 3, 3, 3]
    for mb in minibatches
        @test mb isa NamedTuple{(:q, :p)}
        @test eltype(mb.q) === T
        @test mb.p == 3 .* mb.q
    end
end

# K8: one sample is a loader the training loop can draw from.
@testset "one sample is one minibatch" begin
    x = Float32.(reshape(1:2, 2, 1))
    dl = DataLoader(x, x; suppress_info = true)
    batch = Batch(1)
    minibatches = batch(dl)
    @test length(minibatches) == 1
    @test only(minibatches) == [(1, 1)]
    input, output = only(gathered(dl, batch))
    @test size(input) == (2, 1, 1)
    @test output == input
end

# Edge 3: one count, `cld(n · n_params, batch_size)` with `n` the number of start positions, and an
# `ArgumentError` where no window fits — instead of the assertion and the empty epoch before it.
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
end

# Edges 4 and 5: the classifier's loader, and `Batch(batch_size, dl)` — one draw per sample with the
# whole sequence and the whole output, which is what GMLDatasets calls.
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
    @test length(Batch(2)(dl)) == 5
    for indices in minibatches
        input_batch, output_batch = convert_input_and_batch_indices_to_array(dl, batch, indices)
        @test size(input_batch) == (9, 4, 2)
        @test size(output_batch) == (10, 1, 2)
        @test eltype(input_batch) === T
        @test eltype(output_batch) === Int
        @test output_batch == output[:, :, [i[2] for i in indices]]
    end
end

# Edge 2: the target is the output when there is one, whatever kind the loader is typed as. A
# `DataLoader{T, A3, Array{Int, 3}, :TimeSeries}` is what GMLDatasets and two tests build by naming
# the type, and it must batch like the `:RegularData` loader of the same data.
@testset "a loader with an output is samples whatever its kind" begin
    input = rand(Float32, 9, 4, 10)
    output = zeros(Int, 10, 1, 10)
    for (k, c) in pairs(rand(1:10, 10))
        output[c, 1, k] = 1
    end
    dl = DataLoader(input, output; suppress_info = true)
    dl_time_series = DataLoader{Float32, typeof(input), typeof(output), :TimeSeries}(
        input, output, 9, 4, 10, 10, 1)

    for batch in (Batch(2, dl), Batch(3, 1, 1))
        Random.seed!(11)
        asked = batch(dl_time_series)
        Random.seed!(11)
        reference = batch(dl)
        @test asked == reference
        for (i, j) in zip(asked, reference)
            @test convert_input_and_batch_indices_to_array(dl_time_series, batch, i) ==
                  convert_input_and_batch_indices_to_array(dl, batch, j)
        end
    end
end
