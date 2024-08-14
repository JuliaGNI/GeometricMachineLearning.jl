using CUDA
using GeometricProblems.TodaLattice: hodeensemble
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricMachineLearning
using JLD2

backend = CUDABackend()

params = [(α = α̃ ^ 2, N = 200) for α̃ in 0.8 : .1 : 0.8]
const pr = hodeensemble(; tspan = (0.0, 800.), parameters = params)
const sol = integrate(pr, ImplicitMidpoint())
const dl_cpu_64 = DataLoader(sol; autoencoder = true)
const dl = DataLoader(dl_cpu_64, backend, Float32)

const reduced_dim = 2

const sae_arch = SymplecticAutoencoder(dl.input_dim, reduced_dim; n_encoder_blocks = 4, n_decoder_blocks = 4, n_encoder_layers = 2, n_decoder_layers = 2)
const sae_parameters = load("sae_parameters.jld2")["sae_parameters"] |> cu
const sae_nn = NeuralNetwork(sae_arch, Chain(sae_arch), sae_parameters, backend)

const integrator_train_epochs = 65536
const integrator_batch_size = 4096

const seq_length = 4
const integrator_architecture = StandardTransformerIntegrator(reduced_dim; transformer_dim = 20, n_blocks = 3, n_heads = 5, L = 3, upscaling_activation = tanh)
const integrator_nn = NeuralNetwork(integrator_architecture, backend)
const integrator_method = AdamOptimizerWithDecay(integrator_train_epochs)
const o_integrator = Optimizer(integrator_method, integrator_nn)

loss = GeometricMachineLearning.ReducedLoss(encoder(sae_nn), decoder(sae_nn))

# map autoencoder-like data to time-series like data
dl_integration = DataLoader(dl; autoencoder = false)

# the regular transformer can't deal with symplectic data!
dl_integration = DataLoader(vcat(dl_integration.input.q, dl_integration.input.p))
integrator_batch = Batch(integrator_batch_size, seq_length)
train_integrator_loss = o_integrator(integrator_nn, dl_integration, integrator_batch, integrator_train_epochs, loss)

const mtc = GeometricMachineLearning.map_to_cpu
save("integrator_parameters.jld2", "integrator_parameters", integrator_nn.params |> mtc)