using GeometricProblems.DoublePendulum: DEFAULT_TIMESPAN, DEFAULT_TIMESTEP,
                                        default_parameters,
                                        hodeproblem
using GeometricEquations: EnsembleProblem
using GeometricIntegrators: ImplicitMidpoint, integrate
using LaTeXStrings
using CairoMakie
CairoMakie.activate!()
import Random

# `smoke_size(full, smoke)` returns the second when GML_SMOKE is set, which is how the CI
# job runs this script end to end in seconds. See scripts/README.md.
include("../../utilities/smoke.jl")
Random.seed!(123)

morange = RGBf(255 / 256, 127 / 256, 14 / 256)
mred = RGBf(214 / 256, 39 / 256, 40 / 256)
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)

# ensemble problem. Every member is integrated and then written out as its own figure, so the grid
# is what this script costs: 500 members at the full size, four in a smoke run.
const n_q₁ = smoke_size(10, 2)
const n_q₂ = smoke_size(10, 2)
const n_p₂ = smoke_size(5, 1)

initial_conditions = [(q = [π / i, π / j], p = [0.0, π / k])
                      for i in 1:1:n_q₁, j in 1:1:n_q₂, k in 1:1:n_p₂]
initial_conditions = reshape(initial_conditions, length(initial_conditions))

ensemble_problem = EnsembleProblem(hodeproblem().equation,
    (DEFAULT_TIMESPAN[1], DEFAULT_TIMESPAN[2]),
    DEFAULT_TIMESTEP, initial_conditions, default_parameters())

ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())

function make_plot_for_index(index::Integer = 1)

    # const number_time_steps = 90
    function make_validation_plot(; theme = :dark, symplectic = true)
        textcolor = theme == :dark ? :white : :black
        fig = Figure(; backgroundcolor = :transparent)
        ax = Axis(fig[1, 1];
            backgroundcolor = :transparent,
            bottomspinecolor = textcolor,
            topspinecolor = textcolor,
            leftspinecolor = textcolor,
            rightspinecolor = textcolor,
            xtickcolor = textcolor,
            ytickcolor = textcolor,
            xticklabelcolor = textcolor,
            yticklabelcolor = textcolor,
            xlabel = L"t",
            ylabel = L"q_1",
            xlabelcolor = textcolor,
            ylabelcolor = textcolor
        )

        # we use linewidth  = 2
        lines!(ax, ensemble_solution.s[index].q[:, 1]; color = mblue,
            label = "Implicit midpoint", linewidth = 2)
        axislegend(; position = (0.55, 0.75), backgroundcolor = :transparent, labelcolor = textcolor)

        fig, ax
    end

    fig_light, ax_light = make_validation_plot(; theme = :light)
    fig_dark, ax_dark = make_validation_plot(; theme = :dark)

    mkpath("phase_space_samples")
    CairoMakie.save("phase_space_samples/DoublePendulum-Validation_$(index).png", fig_light)
end

for i in 1:length(ensemble_solution.s)
    make_plot_for_index(i)
end
