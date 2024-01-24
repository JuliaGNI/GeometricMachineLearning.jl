using GeometricMachineLearning: DataLoader, SymplecticTransformer, NeuralNetwork, CPU
using GeometricProblems.DoublePendulum: tspan, tstep, default_parameters, hodeproblem
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricEquations: EnsembleProblem

initial_conditions = [
        (q=[π / 4, π / 2], p=[0.0, π / 8]), 
        (q=[π / 4, π / 1], p=[0.0, 0.0  ]),
        (q=[π / 1, π / 2], p=[0.0, π    ])
]

ensemble_problem = EnsembleProblem(hodeproblem().equation, tspan, tstep, initial_conditions, default_parameters)

ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())

dl = DataLoader(ensemble_solution)

arch = SymplecticTransformer(dl)

nn = NeuralNetwork(arch, CPU(), Float64)