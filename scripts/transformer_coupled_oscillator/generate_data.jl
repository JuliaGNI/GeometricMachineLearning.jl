using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems.CoupledHarmonicOscillator: hodeproblem, default_parameters, tspan, tstep, q₀, p₀
using GeometricEquations: EnsembleProblem
using GeometricMachineLearning: DataLoader

const m₁ = default_parameters.m₁
const m₂ = default_parameters.m₂
const k₁ = default_parameters.k₁
const k₂ = default_parameters.k₂
const k = Float64.(0.0:0.1:4)

params_collection = [(m₁ = m₁, m₂ = m₂, k₁ = k₁, k₂ = k₂, k = k_val) for k_val in k]
ensemble_problem = EnsembleProblem(hodeproblem().equation, tspan, tstep, (q = q₀, p = p₀), params_collection)
ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())

dl_nt = DataLoader(ensemble_solution)