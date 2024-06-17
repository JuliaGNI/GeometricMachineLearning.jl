using CUDA
using GeometricIntegrators: integrate, ImplicitMidpoint
using CairoMakie
using GeometricProblems.TodaLattice: hodeproblem
using GeometricMachineLearning 
import Random

backend = CUDABackend()

pr = hodeproblem(; tspan = (0.0, 100.))
sol = integrate(pr, ImplicitMidpoint())
dl_cpu = DataLoader(sol; autoencoder = true)
dl = DataLoader((
                q = reshape(dl_cpu.input.q |> cu, size(dl_cpu.input.q, 1), size(dl_cpu.input.q, 3)), 
                p = reshape(dl_cpu.input.p |> cu, size(dl_cpu.input.p, 1), size(dl_cpu.input.p, 3))); 
                autoencoder = true)

const reduced_dim = 2

psd_arch = PSDArch(dl.input_dim, reduced_dim)
sae_arch = SymplecticAutoencoder(dl.input_dim, reduced_dim; n_encoder_blocks = 4, n_decoder_blocks = 4, n_encoder_layers = 2, n_decoder_layers = 2)

Random.seed!(123)
psd_nn = NeuralNetwork(psd_arch, backend)
sae_nn = NeuralNetwork(sae_arch, backend)

const n_epochs = 512
const batch_size = 512

o = Optimizer(sae_nn, AdamOptimizer())

psd_error = solve!(psd_nn, dl)
sae_error = o(sae_nn, dl, Batch(batch_size), n_epochs)

hline([psd_error]; color = 2, label = "PSD error")
lines!(sae_error; color = 3, label = "SAE error", xlabel = "epoch", ylabel = "training error")

const mtc = GeometricMachineLearning.map_to_cpu
psd_nn_cpu = mtc(psd_nn)
sae_nn_cpu = mtc(sae_nn)
psd_rs = HRedSys(pr, encoder(psd_nn_cpu), decoder(psd_nn_cpu); integrator = ImplicitMidpoint())
sae_rs = HRedSys(pr, encoder(sae_nn_cpu), decoder(sae_nn_cpu); integrator = ImplicitMidpoint())

projection_error(psd_rs)

projection_error(sae_rs)

sol_full = integrate_full_system(psd_rs)
sol_psd_reduced = integrate_reduced_system(psd_rs)
sol_sae_reduced = integrate_reduced_system(sae_rs)

const t_steps = 100
p_val = lines(sol_full.s.q[t_steps], label = "Implicit Midpoint")
lines!(p_val, psd_rs.decoder((q = sol_psd_reduced.s.q[t_steps], p = sol_psd_reduced.s.p[t_steps])).q, label = "PSD")
lines!(p_val, sae_rs.decoder((q = sol_sae_reduced.s.q[t_steps], p = sol_sae_reduced.s.p[t_steps])).q, label = "SAE")

data_unprocessed = encoder(sae_nn)(dl.input)
data_processed = (  q = reshape(data_unprocessed.q, reduced_dim ÷ 2, length(data_unprocessed.q)), 
                    p = reshape(data_unprocessed.p, reduced_dim ÷ 2, length(data_unprocessed.p))
                    )

dl_reduced = DataLoader(data_processed; autoencoder = false)
integrator_batch_size = 512
integrator_train_epochs = 512

integrator_nn = NeuralNetwork(GSympNet(reduced_dim; n_layers = 5), backend)
o_integrator = Optimizer(AdamOptimizer(), integrator_nn)

loss = GeometricMachineLearning.ReducedLoss(encoder(sae_nn), decoder(sae_nn))
dl_integration = DataLoader((q = reshape(dl.input.q, size(dl.input.q, 1), size(dl.input.q, 3)),
                             p = reshape(dl.input.p, size(dl.input.p, 1), size(dl.input.p, 3)));
                            autoencoder = false
                            )

o_integrator(integrator_nn, dl_integration, Batch(integrator_batch_size), integrator_train_epochs, loss)

ics = (q = mtc(dl_reduced.input.q[:, 1]), p = mtc(dl_reduced.input.p[:, 1]))
time_series = iterate(mtc(integrator_nn), ics; n_points = t_steps)
prediction = (q = time_series.q[:, end], p = time_series.p[:, end])
sol = decoder(sae_nn_cpu)(prediction)

lines!(p_val, sol.q; label = "Neural Network Integrator")

save("symplectic_autoencoder_validation.png", p_val)