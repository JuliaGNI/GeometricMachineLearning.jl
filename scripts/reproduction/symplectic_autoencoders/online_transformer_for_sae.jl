using CUDA
using GeometricProblems.TodaLattice: hodeensemble
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricMachineLearning
using JLD2

# `smoke_size(full, smoke)` returns the second when GML_SMOKE is set, which is how the CI
# job runs this script end to end in seconds. See scripts/README.md.
include("../../utilities/smoke.jl")

# The GPU is what this was run on; the smoke run drops to the CPU so that CI can execute it.
backend = smoke_size(CUDABackend(), CPU())

params = [(α = α̃ ^ 2, N = smoke_size(200, 20)) for α̃ in 0.8:0.1:0.8]
const pr = hodeensemble(; timespan = (0.0, smoke_size(800.0, 5.0)), parameters = params)
const sol = integrate(pr, ImplicitMidpoint())
const dl_cpu_64 = DataLoader(sol; autoencoder = true)
const dl = DataLoader(dl_cpu_64, backend, Float32)

const reduced_dim = 2

const sae_arch = SymplecticAutoencoder(dl.input_dim, reduced_dim; n_encoder_blocks = 4,
    n_decoder_blocks = 4, n_encoder_layers = 2, n_decoder_layers = 2)
# `online_sympnet.jl` writes the autoencoder weights this reads, into whichever directory it is
# run from. Stating the dependency here rather than relying on the caller having run it first.
isfile("sae_parameters.jld2") || include("online_sympnet.jl")

# `cu` moves the loaded weights onto the GPU; on the CPU backend there is nothing to move.
const to_backend = backend == CPU() ? identity : cu
const sae_parameters = JLD2.load("sae_parameters.jld2")["sae_parameters"] |> to_backend
const sae_nn = NeuralNetwork(sae_arch, Chain(sae_arch), sae_parameters, backend)

const integrator_train_epochs = smoke_size(65536, 2)
const integrator_batch_size = 4096

const seq_length = 4
const integrator_architecture = StandardTransformerIntegrator(
    reduced_dim; transformer_dim = 20, n_blocks = 3,
    n_heads = 5, L = 3, upscaling_activation = tanh)
# The element type is not optional here. The `DataLoader` above is `Float32`, and a `Float64`
# network makes `ReducedLoss` fall through to the `NetworkLoss` fallback -- its method has one
# type parameter for the input and the output both, so a mixed pair matches nothing, and what
# the fallback raises is `Functor not defined`, not a `MethodError`.
const integrator_nn = NeuralNetwork(integrator_architecture, backend, Float32)
const integrator_pairing = AdamOptimizerWithDecay(integrator_train_epochs)
const o_integrator = Optimizer(integrator_nn; integrator_pairing...)

loss = GeometricMachineLearning.ReducedLoss(encoder(sae_nn), decoder(sae_nn))

# map autoencoder-like data to time-series like data
dl_integration = DataLoader(dl; autoencoder = false)

# the regular transformer can't deal with symplectic data!
dl_integration = DataLoader(vcat(dl_integration.input.q, dl_integration.input.p))
integrator_batch = Batch(integrator_batch_size, seq_length)
train_integrator_loss = o_integrator(
    integrator_nn, dl_integration, integrator_batch, integrator_train_epochs, loss)

const mtc = GeometricMachineLearning.map_to_cpu
JLD2.save("integrator_parameters.jld2", "integrator_parameters", integrator_nn.params |> mtc)
