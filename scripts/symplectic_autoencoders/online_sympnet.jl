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

const n_epochs = 4096
const batch_size = 512

sae_method = AdamOptimizerWithDecay(n_epochs)
o = Optimizer(sae_nn, sae_method)

psd_error = solve!(psd_nn, dl)
sae_error = o(sae_nn, dl, Batch(batch_size), n_epochs)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "epoch", ylabel = "training error")
morange = RGBf(255 / 256, 127 / 256, 14 / 256)
mred = RGBf(214 / 256, 39 / 256, 40 / 256) 
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)

hlines!([psd_error]; color = morange, label = "PSD error")
lines!(sae_error; color = mgreen, label = "SAE error")
const text_color = :black
axislegend(; position = (.82, .75), backgroundcolor = :transparent, color = text_color)
save("compare_errors.png", fig)

const mtc = GeometricMachineLearning.map_to_cpu
psd_nn_cpu = mtc(psd_nn)
sae_nn_cpu = mtc(sae_nn)
psd_rs = HRedSys(pr, encoder(psd_nn_cpu), decoder(psd_nn_cpu); integrator = ImplicitMidpoint())
sae_rs = HRedSys(pr, encoder(sae_nn_cpu), decoder(sae_nn_cpu); integrator = ImplicitMidpoint())

# projection_error(psd_rs)
# 
# projection_error(sae_rs)

######################################################################

# integrate system
sol_full = integrate_full_system(psd_rs)
sol_psd_reduced = integrate_reduced_system(psd_rs)
sol_sae_reduced = integrate_reduced_system(sae_rs)

######################################################################

# train sympnet 

data_unprocessed = encoder(sae_nn)(dl.input)
data_processed = (  q = reshape(data_unprocessed.q, reduced_dim ÷ 2, length(data_unprocessed.q)), 
                    p = reshape(data_unprocessed.p, reduced_dim ÷ 2, length(data_unprocessed.p))
                    )

dl_reduced = DataLoader(data_processed; autoencoder = false)
integrator_train_epochs = 4096
integrator_batch_size = 512

seq_length = 4
integrator_architecture = StandardTransformerIntegrator(reduced_dim; transformer_dim = 30, n_blocks = 4, n_heads = 5, L = 3, upscaling_activation = tanh)
integrator_nn = NeuralNetwork(integrator_architecture, backend)
integrator_method = AdamOptimizerWithDecay(integrator_train_epochs)
o_integrator = Optimizer(integrator_method, integrator_nn)

loss = GeometricMachineLearning.ReducedLoss(encoder(sae_nn), decoder(sae_nn))
dl_integration = DataLoader((q = reshape(dl.input.q, size(dl.input.q, 1), size(dl.input.q, 3)),
                             p = reshape(dl.input.p, size(dl.input.p, 1), size(dl.input.p, 3)));
                            autoencoder = false
                            )

# the regular transformer can't deal with symplectic data!
dl_integration = DataLoader(vcat(dl_integration.input.q, dl_integration.input.p))
o_integrator(integrator_nn, dl_integration, Batch(integrator_batch_size, seq_length), integrator_train_epochs, loss)

const ics_nt = (q = mtc(dl_reduced.input.q[:, 1:seq_length, 1]), p = mtc(dl_reduced.input.p[:, 1:seq_length, 1]))
const ics = vcat(ics_nt, ics_nt)

######################################################################

# plot validation
function plot_validation(t_steps::Integer=100)
    fig_val = Figure()
    ax_val = Axis(fig_val[1, 1])
    lines!(ax_val, sol_full.s.q[t_steps], label = "Implicit Midpoint", color = mblue)
    lines!(ax_val, psd_rs.decoder((q = sol_psd_reduced.s.q[t_steps], p = sol_psd_reduced.s.p[t_steps])).q, 
        label = "PSD", color = morange)
    lines!(ax_val, sae_rs.decoder((q = sol_sae_reduced.s.q[t_steps], p = sol_sae_reduced.s.p[t_steps])).q, 
        label = "SAE", color = mgreen)

    name = "symplectic_autoencoder_validation_" * string(t_steps)
    axislegend(; position = (.82, .75), backgroundcolor = :transparent, color = text_color)
    save(name * ".png", fig_val)

    time_series = iterate(mtc(integrator_nn), ics; n_points = t_steps, prediction_window = seq_length)
    # prediction = (q = time_series.q[:, end], p = time_series.p[:, end])
    prediction = time.series[:, end]
    sol = decoder(sae_nn_cpu)(prediction)

    lines!(ax_val, sol[1:(dl.sys_dim ÷ 2)]; label = "Neural Network Integrator", color = mpurple)

    axislegend(; position = (.82, .75), backgroundcolor = :transparent, color = text_color)
    save(name * "_with_nn_integrator.png", fig_val)
end

for t_steps in (10, 100, 200, 300, 400, 500, 600, 700, 800, 900)
    plot_validation(t_steps)
end