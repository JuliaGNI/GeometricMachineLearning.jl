# The eight architectures of this package on an Apple GPU: a forward pass, a pullback and two epochs
# under `Adam` and under `GradientMethod`, each against a host twin, under `allowscalar(false)`. One
# line per row, marked pass or fail, and the rows are returned.
#
# Run it by hand through a Kaimon session: Metal is unreachable from a sandboxed shell, where
# `Metal.functional()` is `false`. Activate a scratch environment that develops this tree and the
# local GeometricOptimizers, and restart the session after the last edit of `src/`:
#
#     import Pkg
#     Pkg.activate(; temp = true)
#     Pkg.develop(path = "<this repository>")
#     Pkg.develop(path = "<the GeometricOptimizers checkout>")
#     Pkg.add(["Metal", "GPUArraysCore", "KernelAbstractions", "NeuralNetworkParameters", "Zygote"])
#     include("<this repository>/scripts/metal_check.jl")
#     metal_check()
#
# `Float32` throughout, because Metal has no `Float64`. The two families with a manifold weight train
# under `Geodesic()`: `Cayley()` is built on `lu`, which reads an array one element at a time.
#
# The tree's `Manifest.toml` is gitignored and pins the released GeometricOptimizers, so a session
# opened on the checkout loads that release with no warning, and a row that fails there is the
# release's. The first line below prints the version and the directory of the `GeometricOptimizers`
# that this process loaded, and the sweep stops unless the environment tracks a local path for it.

using GeometricMachineLearning
using GeometricMachineLearning: params
using GeometricMachineLearning.GeometricOptimizers
using GPUArraysCore: allowscalar
using KernelAbstractions
using Metal: Metal, MtlArray
using NeuralNetworkParameters: NetworkParameters, flatten, mapstorage, mapparameters
using Random
import Pkg
import Zygote

const T = Float32

# The measured noise floor of the eight rows below plus a tenth: a device reduction adds in a
# different order than the host's BLAS, and the two agree to a few ulps of a `Float32` and not to
# the bit.
const BOUND = 2e-6

const GO_UUID = Base.UUID("fc236c15-5557-4942-aa65-b650f329279e")

println("GeometricOptimizers ", pkgversion(GeometricOptimizers), " at ",
    pkgdir(GeometricOptimizers))
if !Pkg.dependencies()[GO_UUID].is_tracking_path
    error("the environment resolves a released GeometricOptimizers, not the local tree; " *
          "`Pkg.develop` the checkout before running this sweep")
end

const BACKEND = get_backend(MtlArray(zeros(T, 1)))

# A parameter set on the device, leaf by leaf: `changebackend(::MetalBackend, …)` copies through
# `KernelAbstractions.copyto!`, and the leaf constructors below are the same ones
# `test/helpers/device.jl` uses.
to_device(x::AbstractArray) = MtlArray(x)
to_device(x::StiefelManifold) = StiefelManifold(MtlArray(x.A))
to_device(x::SymmetricMatrix) = SymmetricMatrix(MtlArray(x.S), x.n)
to_device(x::SkewSymMatrix) = SkewSymMatrix(MtlArray(x.S), x.n)
to_device(x::StrictlyLowerTriangular) = StrictlyLowerTriangular(MtlArray(x.S), x.n)
to_device(x::StrictlyUpperTriangular) = StrictlyUpperTriangular(MtlArray(x.S), x.n)
to_device(x::NamedTuple) = map(to_device, x)
to_device(ps::NetworkParameters) = mapparameters(to_device, ps)

# The device result's distance from its host twin, over the two levels of a parameter set and over a
# plain array alike.
deviation(device, host) = maximum(abs, Array(device) .- host)
function deviation(device::NetworkParameters, host)
    maximum(abs,
        first(flatten(mapstorage(Array, device))) .- first(flatten(host)))
end

function check_row(name, f)
    status = try
        f() ? :pass : :wrong
    catch err
        first(split(sprint(showerror, err), '\n'))
    end
    name => status
end

# The two twins of one architecture, built from one draw: the host network, and the same network with
# every leaf of its parameter set on the device.
function twins(arch)
    Random.seed!(7)
    host = NeuralNetwork(arch, CPU(), T)
    Random.seed!(7)
    device = NeuralNetwork(host.architecture, host.model, to_device(params(host)), BACKEND)
    host, device
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
            () -> deviation(host.model(x_device, ps_device), host.model(x, ps_host)) <=
                  BOUND))
    push!(rows,
        check_row("$name, pullback",
            () -> let g_device = Zygote.gradient(p -> sum(abs2, host.model(x_device, p)),
                    ps_device)[1],
                g_host = Zygote.gradient(p -> sum(abs2, host.model(x, p)), ps_host)[1]

                deviation(g_device, g_host) <= BOUND
            end))
    for algorithm in (Adam(), GradientMethod())
        push!(rows,
            check_row("$name, two epochs of $(nameof(typeof(algorithm)))",
                () -> let
                    Random.seed!(11)
                    o_host = TrainingOptimizer(ps_host; algorithm, retraction)
                    h_host = train!(host, o_host,
                        DataLoader(x; suppress_info = true, autoencoder), batch, 2;
                        show_progress = false, loss_kw...)
                    # The device twin draws its global section from the host's stream when its
                    # backend has no generator of its own; a device with one reseeds it here, so the
                    # two twins draw the same section and the row compares one run with another.
                    Random.seed!(11)
                    try
                        Random.seed!(Metal.default_rng(), 11)
                    catch
                    end
                    o_device = TrainingOptimizer(ps_device; algorithm, retraction)
                    h_device = train!(device, o_device,
                        DataLoader(x_device; suppress_info = true, autoencoder), batch, 2;
                        show_progress = false, loss_kw...)
                    deviation(h_device, h_host) <= BOUND
                end))
    end
    rows
end

function metal_check()
    allowscalar(false)
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
    append!(rows,
        network_rows("SymplecticAutoencoder", SymplecticAutoencoder(8, 2),
            randn(T, 8, 12, 4), Batch(6); retraction = Geodesic(), autoencoder = true))
    append!(rows,
        network_rows("StandardTransformerIntegrator", StandardTransformerIntegrator(4),
            randn(T, 4, 8, 12), Batch(6, 3)))
    append!(rows,
        network_rows("VolumePreservingTransformer", VolumePreservingTransformer(4, 3),
            randn(T, 4, 8, 12), Batch(6, 3)))
    for (name, status) in rows
        println(status === :pass ? "PASS  " : "FAIL  ", name,
            status === :pass ? "" : "  ($(status))")
    end
    rows
end
