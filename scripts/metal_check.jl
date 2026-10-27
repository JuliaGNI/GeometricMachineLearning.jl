# The eight architectures of this package on an Apple GPU: a forward pass, a pullback and two epochs
# under `Adam` and under `GradientMethod`, each against a host twin, under `allowscalar(false)`. One
# line per row, with the distance it measured, marked pass or fail, and the rows are returned.
#
# Run it by hand in a Julia session that reaches the GPU: in a sandboxed shell
# `Metal.functional()` is `false`. Activate a scratch environment that develops this tree and the
# local GeometricOptimizers, and restart the session after the last edit of `src/`:
#
#     import Pkg
#     Pkg.activate(; temp = true)
#     Pkg.develop(path = "<this repository>")
#     Pkg.develop(path = "<the GeometricOptimizers checkout>")
#     Pkg.add(["Metal", "GPUArraysCore", "KernelAbstractions", "NeuralNetworkParameters",
#              "AbstractNeuralNetworks", "Zygote"])
#     include("<this repository>/scripts/metal_check.jl")
#     metal_check()
#
# `Float32` throughout, because Metal has no `Float64`. The two families with a manifold weight train
# under `Geodesic()`: `Cayley()` is built on `lu`, which reads an array one element at a time.
#
# The tree's `Manifest.toml` is gitignored and pins the released GeometricOptimizers, so a session
# opened on the checkout loads that release with no warning, and a row that fails there is the
# release's. The first line below prints the version and the directory of the `GeometricOptimizers`
# this process loaded, and the sweep stops unless the environment tracks a path for it and that
# directory is the one it tracks.

using GeometricMachineLearning
import GeometricMachineLearning: GeometricOptimizers, params
using GeometricMachineLearning.GeometricOptimizers
using GPUArraysCore: allowscalar
using KernelAbstractions
using Metal: Metal, MtlArray
using NeuralNetworkParameters: NetworkParameters, flatten, mapstorage
using AbstractNeuralNetworks: changebackend
using Random
import Pkg
import Zygote

const T = Float32

# The distance a row is held to, in units of its own largest host entry: a device result and its host
# twin agree to a few ulps of that entry and not to the bit, and a `Float32` entry of `1e5` has an ulp
# of `0.008`, which no absolute bound below it can hold. The eight architectures differ by four orders
# of magnitude in the entries they compare — the largest entry of `LASympNet`'s pullback is `1e5` — so
# the distance is taken in each row's units: `BOUND` is about 17 `eps(Float32)` of the row's largest
# host magnitude, and for a row whose values are `O(1)` it is that absolute bound. Measured here, the
# worst row is `3e-7`.
const BOUND = 2e-6

const GO_UUID = Base.UUID("fc236c15-5557-4942-aa65-b650f329279e")

const GO_DIR = pkgdir(GeometricOptimizers)
println("GeometricOptimizers ", pkgversion(GeometricOptimizers), " at ", GO_DIR)

# The rows measure the module this process loaded, and the environment says only which directory that
# must be. Two things hold before a row runs, each with a stop of its own: the environment tracks a
# path for `GeometricOptimizers`, so that it is a checkout and not a release, and the module this
# process loaded came from that path, because a package is loaded once per process and a session that
# loaded the release keeps it through `Pkg.develop`.
const GO_DEP = Pkg.dependencies()[GO_UUID]
if !GO_DEP.is_tracking_path
    error(
        "the environment does not track a path for GeometricOptimizers: it resolves the release ",
        GO_DEP.source, ". A manifest that pins the release, as this tree's gitignored ",
        "`Manifest.toml` does, resolves it; `Pkg.develop` the local checkout instead")
elseif abspath(GO_DIR) != abspath(GO_DEP.source)
    error("the loaded GeometricOptimizers is at ", GO_DIR, ", and the environment tracks ",
        GO_DEP.source, ". A session that loaded a released GeometricOptimizers keeps that ",
        "module through `Pkg.develop`; restart the session so that the sweep measures the tree ",
        "the environment names")
end

const BACKEND = get_backend(MtlArray(zeros(T, 1)))

# The distance of a device result from its host twin, over the two levels of a parameter set and over a
# plain array alike.
function deviation(device, host)
    maximum(abs, Array(device) .- host) / max(one(T), maximum(abs, host))
end
function deviation(device::NetworkParameters, host)
    d, h = first(flatten(mapstorage(Array, device))), first(flatten(host))
    maximum(abs, d .- h) / max(one(T), maximum(abs, h))
end

function check_row(name, f)
    status = :pass
    detail = try
        d = f()
        status = d <= BOUND ? :pass : :wrong
        string("distance ", d)
    catch err
        status = :wrong
        first(split(sprint(showerror, err), '\n'))
    end
    println(status === :pass ? "PASS  " : "FAIL  ", name, "  (", detail, ")")
    name => status
end

# The two twins of one architecture: the host network, and the same network with every leaf of its
# parameter set on the device. `changebackend` hands each leaf's storage to the device and rebuilds the
# leaf around it, so a `StiefelManifold`, a `SymmetricMatrix`, a `SkewSymMatrix`, both triangular types
# and the `NamedTuple` of a transformer's attention heads keep their shape; the device set is a copy of
# the host's, so the two start from one draw.
function twins(arch)
    Random.seed!(7)
    host = NeuralNetwork(arch, CPU(), T)
    device = NeuralNetwork(host.architecture, host.model,
        changebackend(BACKEND, params(host)), BACKEND)
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

# The rows of one architecture. `x` is the host input, `batch` the minibatch of its family, and
# `retraction` is `Geodesic()` for the two families with a manifold weight.
function network_rows(name, arch, x, batch; loss = nothing, retraction = Cayley(),
        autoencoder = false)
    host, device = twins(arch)
    x_device = MtlArray(x)
    ps_host, ps_device = params(host), params(device)
    loss_kw = loss === nothing ? (;) : (; loss)

    rows = Pair{String, Any}[]
    push!(rows,
        check_row("$name, forward",
            () -> deviation(host.model(x_device, ps_device), host.model(x, ps_host))))
    push!(rows,
        check_row("$name, pullback",
            () -> let g_device = Zygote.gradient(p -> sum(abs2, host.model(x_device, p)),
                    ps_device)[1],
                g_host = Zygote.gradient(p -> sum(abs2, host.model(x, p)), ps_host)[1]

                deviation(g_device, g_host)
            end))
    for algorithm in (Adam(), GradientMethod())
        push!(rows,
            check_row("$name, two epochs of $(nameof(typeof(algorithm)))",
                () -> let
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
                    deviation(h_device, h_host)
                end))
    end
    rows
end

function metal_check()
    allowscalar(false)
    # One seed for the sweep, so that a run repeats the distances of the run that wrote the comment on
    # `BOUND`: every row compares two twins of one draw, and the question a reader asks of the sweep is
    # whether these eight rows still hold.
    Random.seed!(7)
    rows = Pair{String, Any}[]
    append!(rows,
        network_rows("a Dense chain", Chain(Dense(4, 8, tanh), Dense(8, 4)),
            randn(T, 4, 12, 4), Batch(6); loss = FeedForwardLoss()))
    append!(rows,
        network_rows("a StiefelLayer chain",
            Chain(StiefelLayer(4, 8), Dense(8, 4, tanh)), randn(T, 4, 12, 4), Batch(6);
            loss = FeedForwardLoss(), retraction = Geodesic()))
    append!(rows, network_rows("GSympNet", GSympNet(4), randn(T, 4, 12, 4), Batch(6)))
    append!(rows, network_rows("LASympNet", LASympNet(4), randn(T, 4, 12, 4), Batch(6)))
    append!(rows,
        network_rows("VolumePreservingFeedForward", VolumePreservingFeedForward(4),
            randn(T, 4, 12, 4), Batch(6)))
    # A latent dimension of four and not two: at two, two of this parameter set's `StiefelManifold`
    # weights hold a single column, whose horizontal lift has an empty skew block, and a Metal launch of
    # an empty array raises `DivideError` (`KNOWN_ISSUES.md`, K34).
    append!(rows,
        network_rows("SymplecticAutoencoder", SymplecticAutoencoder(8, 4),
            randn(T, 8, 12, 4), Batch(6); retraction = Geodesic(), autoencoder = true))
    append!(rows,
        network_rows("StandardTransformerIntegrator", StandardTransformerIntegrator(4),
            randn(T, 4, 8, 12), Batch(6, 3)))
    append!(rows,
        network_rows("VolumePreservingTransformer", VolumePreservingTransformer(4, 3),
            randn(T, 4, 8, 12), Batch(6, 3)))
    failures = count(row -> last(row) !== :pass, rows)
    println()
    println(length(rows), " rows, ", failures, " not passing")
    rows
end
