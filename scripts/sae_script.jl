using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricMachineLearning 
import Random # hide
import GeometricProblems.TodaLattice as tl
using JLD2


N = tl.Ñ # hide
Δx = 1. / (N - 1) # hide
Ω = -0.5 : Δx : 0.5 # hide
tl.μ

pr = tl.hodeproblem(; tspan = (0.0, 800.))
sol = integrate(pr, ImplicitMidpoint())

dl_cpu = DataLoader(sol; autoencoder = true, suppress_info = true)

const reduced_dim = 2

Random.seed!(123) # hide
sae_arch = SymplecticAutoencoder(dl_cpu.input_dim, reduced_dim; n_encoder_blocks = 4, 
                                                                n_decoder_blocks = 4, 
                                                                n_encoder_layers = 2, 
                                                                n_decoder_layers = 2)



const mtc = GeometricMachineLearning.map_to_cpu

sae_trained_parameters = load("../docs/src/tutorials/sae_parameters.jld2")["sae_parameters"]
_nnp(ps::Tuple) = NeuralNetworkParameters{Tuple(Symbol("L$(i)") for i in 1:length(ps))}(ps)
sae_nn_cpu = NeuralNetwork(sae_arch, Chain(sae_arch), _nnp(sae_trained_parameters), CPU())

sae_rs = HRedSys(pr, encoder(sae_nn_cpu), decoder(sae_nn_cpu); integrator = ImplicitMidpoint())

nothing  # hide

@time "FOM + Implicit Midpoint" sol_full = integrate_full_system(sae_rs) # hide
@time "SAE + Implicit Midpoint" sol_sae_reduced = integrate_reduced_system(sae_rs) # hide

backend = CPU() # hide
const integrator_train_epochs = 65536
const integrator_batch_size = 4096
const seq_length = 4

integrator_architecture = StandardTransformerIntegrator(reduced_dim; 
                                                                    transformer_dim = 20, 
                                                                    n_blocks = 3, 
                                                                    n_heads = 5, 
                                                                    L = 3,
                                                                    upscaling_activation = tanh)

nn_integrator_parameters = load("../docs/src/tutorials/integrator_parameters.jld2")["integrator_parameters"] # hide
integrator_nn = NeuralNetwork(integrator_architecture, Chain(integrator_architecture), _nnp(nn_integrator_parameters), backend) # hide
ics = encoder(sae_nn_cpu)((q = dl.input.q[:, 1:seq_length, 1], p = dl.input.p[:, 1:seq_length, 1])) # hide
iterate(mtc(integrator_nn), ics; n_points = length(sol.t), prediction_window = seq_length) # hide
@time "time stepping with transformer" time_series = iterate(mtc(integrator_nn), ics; n_points = length(sol.t), prediction_window = seq_length)

morange = RGBf(255 / 256, 127 / 256, 14 / 256)
mred = RGBf(214 / 256, 39 / 256, 40 / 256) 
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)

time_series = iterate(mtc(integrator_nn), ics; n_points = length(sol.t) * 10, prediction_window = seq_length)
time_steps = 1:10:1000 # axes(time_series.q, 2)
function plot_solution(time_step; theme = :light, framerate = 50)
    textcolor = theme == :dark ? :white : :black 
    fig = Figure()
    ax = Axis(fig[1, 1],    backgroundcolor = :transparent,
                            bottomspinecolor = textcolor, 
                            topspinecolor = textcolor,
                            leftspinecolor = textcolor,
                            rightspinecolor = textcolor,
                            xtickcolor = textcolor, 
                            ytickcolor = textcolor,
                            xticklabelcolor = textcolor,
                            yticklabelcolor = textcolor,
                            xlabel=L"\omega", 
                            ylabel=L"q",
                            xlabelcolor = textcolor,
                            ylabelcolor = textcolor)
    lines!(ax, sol_full.s.q[time_step], label = rich("FOM + Implicit Midpoint"; color = textcolor), color = mblue)
    lines!(ax, sae_rs.decoder((q = sol_sae_reduced.s.q[time_step], p = sol_sae_reduced.s.p[time_step])).q, label = rich("SAE + Implicit Midpoint"; color = textcolor), color = mgreen)
    fig
end

function plot_solution2(time_step; theme = :light, framerate = 50)
    textcolor = theme == :dark ? :white : :black 
    fig = Figure()
    ax = Axis(fig[1, 1],    backgroundcolor = :transparent,
                            bottomspinecolor = textcolor, 
                            topspinecolor = textcolor,
                            leftspinecolor = textcolor,
                            rightspinecolor = textcolor,
                            xtickcolor = textcolor, 
                            ytickcolor = textcolor,
                            xticklabelcolor = textcolor,
                            yticklabelcolor = textcolor,
                            xlabel=L"\omega", 
                            ylabel=L"q",
                            xlabelcolor = textcolor,
                            ylabelcolor = textcolor)
    lines!(ax, sol_full.s.q[time_step], label = rich("FOM + Implicit Midpoint"; color = textcolor), color = mblue)
    time_series = iterate(mtc(integrator_nn), ics; n_points = time_step, prediction_window = seq_length)
    # prediction = (q = time_series.q[:, end], p = time_series.p[:, end])
    prediction = (q = time_series.q[:, end], p = time_series.p[:, end])
    sol = decoder(sae_nn_cpu)(prediction)

    lines!(ax_val, Ω, sol.q; label = rich("SAE + Transformer"; color = textcolor), color = mpurple)
    fig
end

sae_dir = "animations"
for time_step in time_steps
    fig = plot_solution(time_step)
    save(sae_dir * "/sae-$(string(time_step, pad = 3)).pdf", fig)

    fig2 = plot_solution2(time_step)
    save(sae_dir * "/transformer-$(string(time_step, pad = 3)).pdf", fig2)
end