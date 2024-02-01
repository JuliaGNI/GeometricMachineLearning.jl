using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems.CoupledHarmonicOscillator: hodeproblem, default_parameters, tspan, tstep, q₀, p₀
using GeometricEquations: EnsembleProblem
using GeometricMachineLearning: DataLoader
using Plots

const m₁ = default_parameters.m₁
const m₂ = default_parameters.m₂
const k₁ = default_parameters.k₁
const k₂ = default_parameters.k₂
const k = [0.0, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0]

params_collection = [(m₁ = m₁, m₂ = m₂, k₁ = k₁, k₂ = k₂, k = k_val) for k_val in k]
ensemble_problem = EnsembleProblem(hodeproblem().equation, tspan, tstep, (q = q₀, p = p₀), params_collection)
ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())

dl = DataLoader(ensemble_solution)

function plot_curves(dl::DataLoader, one_plot=true, psize=(1500,1000))
    q₁ = dl.input.q[1, :, :]
    q₂ = dl.input.q[2, :, :]
    p₁ = dl.input.p[1, :, :]
    p₂ = dl.input.p[2, :, :]
    h = tspan[2] / (size(q₁, 1) - 1)
    t = 0.0:h:tspan[2]
    n_param_sets = length(params_collection)
    labels = reshape(["k = "*string(params.k) for params in params_collection], 1, n_param_sets)
    plot_q₁ = one_plot ? plot(t, q₁, size=psize) : plot(t, q₁, layout=(n_param_sets, 1), size=psize, label=labels, legend=:topright)
    plot_q₂ = one_plot ? plot(t, q₂, size=psize) : plot(t, q₂, layout=(n_param_sets, 1), size=psize, label=labels, legend=:topright)
    plot_p₁ = one_plot ? plot(t, p₁, size=psize) : plot(t, p₁, layout=(n_param_sets, 1), size=psize, label=labels, legend=:topright)
    plot_p₂ = one_plot ? plot(t, p₂, size=psize) : plot(t, p₁, layout=(n_param_sets, 1), size=psize, label=labels, legend=:topright)
    png(plot_q₁, "plots_of_data_set/q1")
    png(plot_q₂, "plots_of_data_set/q2")
    png(plot_p₁, "plots_of_data_set/p1")
    png(plot_p₂, "plots_of_data_set/p2")
end

plot_curves(dl, false)

