# The eight architectures of this package on an Apple GPU: a forward pass, a pullback and two epochs
# under `Adam` and under `GradientMethod`, each against a host twin, under `allowscalar(false)`. Each
# row is one testset, which asserts the distance it measured. `runtests.jl` includes this file in the
# `metal` group, beside `test/devices/metal.jl`, and like that file it records one skip,
# `@test_skip Metal.functional()`, where `Metal.functional()` is false.
#
# `Float32` throughout, because Metal has no `Float64`. The two families with a manifold weight train
# under `Geodesic()`: `Cayley()` is built on `lu`, which reads an array one element at a time.
#
# The rows measure the `GeometricOptimizers` this process loaded. The file prints its version and its
# directory before the first row, and warns where that directory is a registered release in a depot
# and not a checkout, so that a row that fails is read against the code that ran it.

using GeometricMachineLearning
import GeometricMachineLearning: GeometricOptimizers, params
using GeometricMachineLearning.GeometricOptimizers
using GPUArraysCore: allowscalar
using KernelAbstractions
using Metal: Metal, MtlArray
using NeuralNetworkParameters: NetworkParameters, flatten, mapstorage
using AbstractNeuralNetworks: changebackend
using Random
using Test
import Zygote

const T = Float32

# The distance a row is held to, in units of its own largest host entry: a device result and its host
# twin agree to a few ulps of that entry and not to the bit, and a `Float32` entry of `1e5` has an ulp
# of `0.008`, which no absolute bound below it can hold. The eight architectures differ by four orders
# of magnitude in the entries they compare — the largest entry of `LASympNet`'s pullback is `1e5` — so
# the distance is taken in each row's units: `BOUND` is about 17 `eps(Float32)` of the row's largest
# host magnitude, and for a row whose values are `O(1)` it is that absolute bound. Measured on an Apple
# M4 Max, the worst row is `3e-7`.
const BOUND = 2e-6

# The distance of a device result from its host twin, over the two levels of a parameter set and over a
# plain array alike.
function deviation(device, host)
    maximum(abs, Array(device) .- host) / max(one(T), maximum(abs, host))
end
function deviation(device::NetworkParameters, host)
    d, h = first(flatten(mapstorage(Array, device))), first(flatten(host))
    maximum(abs, d .- h) / max(one(T), maximum(abs, h))
end

# The two twins of one architecture: the host network, and the same network with every leaf of its
# parameter set on the device. `changebackend` hands each leaf's storage to the device and rebuilds the
# leaf around it, so a `StiefelManifold`, a `SymmetricMatrix`, a `SkewSymMatrix`, both triangular types
# and the `NamedTuple` of a transformer's attention heads keep their shape; the device set is a copy of
# the host's, so the two start from one draw.
function twins(arch, backend)
    Random.seed!(7)
    host = NeuralNetwork(arch, CPU(), T)
    device = NeuralNetwork(host.architecture, host.model,
        changebackend(backend, params(host)), backend)
    host, device
end

# Two epochs of one optimizer, an epoch per call and the stream reseeded before each: an epoch of a
# device network advances the CPU random stream further than the host's does, so one `train!` of two
# epochs would hand the twins different minibatches from the second epoch on. `train!` with the same
# optimizer resumes where the previous call stopped, so the two calls are the two epochs of one run.
function two_epochs!(nn, opt, dl, batch, loss_kw)
    reduce(vcat, [begin
                      Random.seed!(11)
                      train!(nn, opt, dl, batch, 1; show_progress = false, loss_kw...)
                  end
                  for _ in 1:2])
end

# A `GlobalSection` holds the random completion of a manifold leaf's frame, and the two twins draw it
# from different generators: the host's from the CPU stream, the device's from the GPU's. Measured on
# the `StiefelLayer` chain, the two completions of its one manifold weight differ by 0.95 entry by
# entry, and the twins then retract along different geodesics. `copyto!` moves a section onto a
# destination with another array type, so the host's sections are moved onto the device before the
# twins run and both start from one frame.
function share_sections!(dest, src)
    if src isa NamedTuple
        for k in keys(src)
            share_sections!(dest[k], src[k])
        end
    else
        copyto!(dest, src)
    end
    dest
end

# The rows of one architecture, each a testset of its own, so that a row that raises is recorded as an
# error and the rows after it still run. `x` is the host input, `batch` the minibatch of its family,
# and `retraction` is `Geodesic()` for the two families with a manifold weight.
function network_rows(name, arch, x, batch, backend; loss = nothing, retraction = Cayley(),
        autoencoder = false)
    host, device = twins(arch, backend)
    x_device = MtlArray(x)
    ps_host, ps_device = params(host), params(device)
    loss_kw = loss === nothing ? (;) : (; loss)

    @testset "$name" begin
        @testset "forward" begin
            @test deviation(host.model(x_device, ps_device), host.model(x, ps_host)) <=
                  BOUND
        end
        @testset "pullback" begin
            g_device = Zygote.gradient(p -> sum(abs2, host.model(x_device, p)), ps_device)[1]
            g_host = Zygote.gradient(p -> sum(abs2, host.model(x, p)), ps_host)[1]
            @test deviation(g_device, g_host) <= BOUND
        end
        @testset "two epochs of $(nameof(typeof(algorithm)))" for algorithm in (Adam(), GradientMethod())
            Random.seed!(11)
            o_host = TrainingOptimizer(ps_host; algorithm, retraction)
            o_device = TrainingOptimizer(ps_device; algorithm, retraction)
            share_sections!(GeometricOptimizers.section(o_device.cache),
                GeometricOptimizers.section(o_host.cache))
            share_sections!(GeometricOptimizers.section(o_device.state),
                GeometricOptimizers.section(o_host.state))
            h_host = two_epochs!(host, o_host,
                DataLoader(x; suppress_info = true, autoencoder), batch, loss_kw)
            h_device = two_epochs!(device, o_device,
                DataLoader(x_device; suppress_info = true, autoencoder), batch, loss_kw)
            @test deviation(h_device, h_host) <= BOUND
        end
    end
end

if Metal.functional()
    go_dir = pkgdir(GeometricOptimizers)
    println("GeometricOptimizers ", pkgversion(GeometricOptimizers), " at ", go_dir)
    if any(depot -> startswith(go_dir, joinpath(abspath(depot), "packages")), DEPOT_PATH)
        @warn "The loaded GeometricOptimizers is a registered release and not a checkout, so the rows measure that release." go_dir
    end

    allowscalar(false)
    backend = get_backend(MtlArray(zeros(T, 1)))
    # One seed for the sweep, so that a run repeats the distances of the run that wrote the comment on
    # `BOUND`: every row compares two twins of one draw.
    Random.seed!(7)
    network_rows("a Dense chain", Chain(Dense(4, 8, tanh), Dense(8, 4)),
        randn(T, 4, 12, 4), Batch(6), backend; loss = FeedForwardLoss())
    network_rows("a StiefelLayer chain",
        Chain(StiefelLayer(4, 8), Dense(8, 4, tanh)), randn(T, 4, 12, 4), Batch(6), backend;
        loss = FeedForwardLoss(), retraction = Geodesic())
    network_rows("GSympNet", GSympNet(4), randn(T, 4, 12, 4), Batch(6), backend)
    network_rows("LASympNet", LASympNet(4), randn(T, 4, 12, 4), Batch(6), backend)
    network_rows("VolumePreservingFeedForward", VolumePreservingFeedForward(4),
        randn(T, 4, 12, 4), Batch(6), backend)
    # A latent dimension of four and not two: at two, two of this parameter set's `StiefelManifold`
    # weights hold a single column, whose horizontal lift has an empty skew block, and a Metal launch of
    # an empty array raises `DivideError` (`KNOWN_ISSUES.md`, K34).
    network_rows("SymplecticAutoencoder", SymplecticAutoencoder(8, 4),
        randn(T, 8, 12, 4), Batch(6), backend; retraction = Geodesic(), autoencoder = true)
    network_rows("StandardTransformerIntegrator", StandardTransformerIntegrator(4),
        randn(T, 4, 8, 12), Batch(6, 3), backend)
    network_rows("VolumePreservingTransformer", VolumePreservingTransformer(4, 3),
        randn(T, 4, 8, 12), Batch(6, 3), backend)
else
    @test_skip Metal.functional()
end
