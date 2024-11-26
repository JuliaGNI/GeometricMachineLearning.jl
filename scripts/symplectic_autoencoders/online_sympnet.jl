using CUDA
using GeometricIntegrators: integrate, ImplicitMidpoint
using CairoMakie
using GeometricProblems.TodaLattice: hodeensemble
using GeometricMachineLearning 
using JLD2
import Random

backend = CUDABackend()

params = [(α = α̃ ^ 2, N = 200) for α̃ in 0.8 : .1 : 0.8]
pr = hodeensemble(; tspan = (0.0, 800.), parameters = params)
sol = integrate(pr, ImplicitMidpoint())
dl_cpu_64 = DataLoader(sol; autoencoder = true)
dl = DataLoader(dl_cpu_64, backend, Float32)

const reduced_dim = 2

psd_arch = PSDArch(dl.input_dim, reduced_dim)
sae_arch = SymplecticAutoencoder(dl.input_dim, reduced_dim; n_encoder_blocks = 4, n_decoder_blocks = 4, n_encoder_layers = 2, n_decoder_layers = 2)

Random.seed!(123)
psd_nn = NeuralNetwork(psd_arch, backend)
sae_nn = NeuralNetwork(sae_arch, backend)

const n_epochs = 262144
const batch_size = 4096

sae_method = AdamOptimizerWithDecay(n_epochs)
o = Optimizer(sae_nn, sae_method)

println("Number of batches: ", GeometricMachineLearning.number_of_batches(dl, Batch(batch_size)))

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
save("symplectic_autoencoder_validation/compare_errors.pdf", fig)

const mtc = GeometricMachineLearning.map_to_cpu

save("sae_parameters.jld2", "sae_parameters", sae_nn.params |> mtc, "training loss", sae_error)

psd_nn_cpu = mtc(psd_nn)
sae_nn_cpu = mtc(sae_nn)
psd_rs = HRedSys(pr, encoder(psd_nn_cpu), decoder(psd_nn_cpu); integrator = ImplicitMidpoint())
sae_rs = HRedSys(pr, encoder(sae_nn_cpu), decoder(sae_nn_cpu); integrator = ImplicitMidpoint())

# projection_error(psd_rs)
# 
# projection_error(sae_rs)

######################################################################

# integrate system
@time "FOM + Implicit Midpoint" sol_full = integrate_full_system(psd_rs)
@time "PSD + Implicit Midpoint" sol_psd_reduced = integrate_reduced_system(psd_rs)
@time "SAE + Implicit Midpoint" sol_sae_reduced = integrate_reduced_system(sae_rs)

# call same functions again.
@time "FOM + Implicit Midpoint" sol_full = integrate_full_system(psd_rs)
@time "PSD + Implicit Midpoint" sol_psd_reduced = integrate_reduced_system(psd_rs)
@time "SAE + Implicit Midpoint" sol_sae_reduced = integrate_reduced_system(sae_rs)

######################################################################

# train sympnet 

data_processed = encoder(sae_nn)(dl.input)

dl_reduced = DataLoader(data_processed; autoencoder = false)
integrator_train_epochs = 262144
integrator_batch_size = 4096

seq_length = 4
integrator_architecture = StandardTransformerIntegrator(reduced_dim; transformer_dim = 10, n_blocks = 3, n_heads = 5, L = 2, upscaling_activation = tanh)
integrator_nn = NeuralNetwork(integrator_architecture, backend)
integrator_method = AdamOptimizerWithDecay(integrator_train_epochs)
o_integrator = Optimizer(integrator_method, integrator_nn)

loss = GeometricMachineLearning.ReducedLoss(encoder(sae_nn), decoder(sae_nn))

# map autoencoder-like data to time-series like data
dl_integration = DataLoader(dl; autoencoder = false)

# the regular transformer can't deal with symplectic data!
dl_integration = DataLoader(vcat(dl_integration.input.q, dl_integration.input.p))
integrator_batch = Batch(integrator_batch_size, seq_length)
train_integrator_loss = o_integrator(integrator_nn, dl_integration, integrator_batch, integrator_train_epochs, loss)

save("integrator_parameters.jld2", "integrator_parameters", integrator_nn.params |> mtc, "training loss", train_integrator_loss)

const ics_nt = (q = mtc(dl_reduced.input.q[:, 1:seq_length, 1]), p = mtc(dl_reduced.input.p[:, 1:seq_length, 1]))
const ics = vcat(ics_nt.q, ics_nt.p)

######################################################################

# plot validation
function plot_validation(t_steps::Integer=100)
    fig_val = Figure()
    ax_val = Axis(fig_val[1, 1])
    lines!(ax_val, sol_full.s.q[t_steps], label = "FOM + Implicit Midpoint", color = mblue)
    lines!(ax_val, psd_rs.decoder((q = sol_psd_reduced.s.q[t_steps], p = sol_psd_reduced.s.p[t_steps])).q, 
        label = "PSD + Implicit Midpoint", color = morange)
    lines!(ax_val, sae_rs.decoder((q = sol_sae_reduced.s.q[t_steps], p = sol_sae_reduced.s.p[t_steps])).q, 
        label = "SAE + Implicit Midpoint", color = mgreen)

    name = "symplectic_autoencoder_validation/symplectic_autoencoder_validation_" * string(t_steps)
    # axislegend(; position = (.82, .75), backgroundcolor = :transparent, color = text_color)
    save(name * ".pdf", fig_val)

    @time "time stepping with transformer" time_series = iterate(mtc(integrator_nn), ics; n_points = t_steps, prediction_window = seq_length)
    # prediction = (q = time_series.q[:, end], p = time_series.p[:, end])
    prediction = time_series[:, end]
    sol = decoder(sae_nn_cpu)(prediction)

    lines!(ax_val, sol[1:(dl.input_dim ÷ 2)]; label = "SAE + Transformer", color = mpurple)

    axislegend(; position = (.82, .75), backgroundcolor = :transparent, color = text_color)
    save(name * "_with_nn_integrator.pdf", fig_val)
end

# for t_steps in 0:20:9980
#     plot_validation(t_steps)
# end

######################################################################

# plot the reduced data (should be a closed orbit)
reduced_data_matrix = vcat(dl_reduced.input.q, dl_reduced.input.p)
fig_reduced = Figure()
ax_reduced = Axis(fig_reduced[1, 1])
lines!(ax_reduced, reduced_data_matrix[1, :, 1] |> mtc, reduced_data_matrix[2, :, 1] |> mtc; color = mgreen, label = "Reduced Data")
axislegend(; position = (.82, .75), backgroundcolor = :transparent, color = text_color)
save("symplectic_autoencoder_validation/reduced_data.pdf", fig_reduced)