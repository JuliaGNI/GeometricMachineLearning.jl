using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricMachineLearning 
import Random # hide
import GeometricProblems.TodaLattice as tl
using JLD2
using CairoMakie

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

# @time "FOM + Implicit Midpoint" sol_full = integrate_full_system(sae_rs) # hide
@time "SAE + Implicit Midpoint" sol_sae_reduced = integrate_reduced_system(sae_rs) # hide

const T = Float32
_T(qp::NamedTuple{(:q, :p)}) = (q = T.(qp.q), p = T.(qp.p))

dl_reduced = DataLoader(encoder(sae_nn_cpu)(_T(dl_cpu.input)))

# lines(dl_reduced.input.q[1, :, 1], dl_reduced.input.p[1, :, 1])

sympnet_arch = GSympNet(2; n_layers = 10)
sympnet_nn = NeuralNetwork(sympnet_arch, T)
o = Optimizer(AdamOptimizer(), sympnet_nn)
o(sympnet_nn, dl_reduced, Batch(10), 500)

morange = RGBf(255 / 256, 127 / 256, 14 / 256)
mred = RGBf(214 / 256, 39 / 256, 40 / 256) 
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)

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
    lines!(ax, sol.s.q[time_step], label = rich("FOM + Implicit Midpoint"; color = textcolor), color = mblue)
    lines!(ax, sae_rs.decoder((q = sol_sae_reduced.s.q[time_step], p = sol_sae_reduced.s.p[time_step])).q, label = rich("SAE + Implicit Midpoint"; color = textcolor), color = mgreen)
    axislegend(ax; position = (1.01, 1.5), labelsize = 8)
    fig
end


#### Transformer 

const seq_length = 4
integrator_architecture = StandardTransformerIntegrator(reduced_dim; 
                                                                    transformer_dim = 20, 
                                                                    n_blocks = 3, 
                                                                    n_heads = 5, 
                                                                    L = 3,
                                                                    upscaling_activation = tanh)

nn_integrator_parameters = load("../docs/src/tutorials/integrator_parameters.jld2")["integrator_parameters"] # hide
integrator_nn = NeuralNetwork(integrator_architecture, Chain(integrator_architecture), _nnp(nn_integrator_parameters), CPU()) # hide

n_time_steps = 10000

ics = (q = dl_reduced.input.q[:, 1:seq_length], p = dl_reduced.input.p[:, 1:seq_length])
time_series = iterate(mtc(integrator_nn), ics; n_points = n_time_steps, prediction_window = seq_length)
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
    lines!(ax, sol.s.q[time_step], label = rich("FOM + Implicit Midpoint"; color = textcolor), color = mblue)
    # prediction = (q = time_series.q[:, end], p = time_series.p[:, end])
    prediction = (q = time_series.q[:, time_step], p = time_series.p[:, time_step])
    prediction_big = decoder(sae_nn_cpu)(prediction)

    lines!(ax, prediction_big.q; label = rich("SAE + Transformer"; color = textcolor), color = mpurple)
    axislegend(ax; position = (1.01, 1.5), labelsize = 8)
    fig
end

ics3 = (q = ics.q[:, 1], p = ics.p[:, 1])

time_series2 = iterate(sympnet_nn, ics3; n_points = n_time_steps)

function plot_solution3(time_step; theme = :light, framerate = 50)
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
    lines!(ax, sol.s.q[time_step], label = rich("FOM + Implicit Midpoint"; color = textcolor), color = mblue)
    time_series = iterate(sympnet_nn, ics3; n_points = time_step)
    # prediction = (q = time_series.q[:, end], p = time_series.p[:, end])
    prediction = (q = time_series2.q[:, time_step], p = time_series2.p[:, time_step])
    prediction_big = decoder(sae_nn_cpu)(prediction)

    lines!(ax, prediction_big.q; label = rich("SAE + SympNet"; color = textcolor), color = mpurple)
    axislegend(ax; position = (1.01, 1.5), labelsize = 8)
    fig
end

time_steps = 1:10:500 # axes(time_series.q, 2)
sae_dir = "animations"
for time_step in time_steps
    fig = plot_solution(time_step)
    save(sae_dir * "/sae-$(string(time_step, pad = 3)).pdf", fig)

    fig2 = plot_solution2(time_step)
    save(sae_dir * "/transformer-$(string(time_step, pad = 3)).pdf", fig2)

    fig3 = plot_solution3(time_step)
    save(sae_dir * "/SympNet-$(string(time_step, pad = 3)).pdf", fig3)
end