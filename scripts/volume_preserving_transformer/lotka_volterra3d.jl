using GeometricMachineLearning
using GeometricMachineLearning: transformer_loss, apply_toNT, map_to_cpu
# using Plots
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems.LotkaVolterra3d: lotka_volterra_3d_ode, default_parameters, tspan, Δt, hamiltonian, X₀, Y₀, Z₀
using GeometricEquations: EnsembleProblem

# hyperparameters for the problem 
const q₀ = [(q₀ = [X₀, Y₀, Z₀], ) for Z₀ in 1. : .1 : 2.]

ensemble_problem = EnsembleProblem(lotka_volterra_3d_ode().equation, tspan, Δt, q₀, default_parameters)
ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())

dl_nt = DataLoader(ensemble_solution)