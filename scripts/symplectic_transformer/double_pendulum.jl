using GeometricMachineLearning: DataLoader, LinearSymplecticTransformer, NeuralNetwork, CPU, Batch, AdamOptimizer, Optimizer, transformer_loss
using GeometricProblems.DoublePendulum: tspan, tstep, default_parameters, hodeproblem
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricEquations: EnsembleProblem

initial_conditions = [
        (q=[π / 4, π / 2], p=[0.0, π / 8]), 
        (q=[π / 4, π / 1], p=[0.0, 0.0  ]),
        (q=[π / 1, π / 2], p=[0.0, π    ])
]

ensemble_problem = EnsembleProblem(hodeproblem().equation, (tspan[1], 10*tspan[2]), tstep, initial_conditions, default_parameters)

ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())

dl = DataLoader(ensemble_solution)

const seq_length = 5

arch = LinearSymplecticTransformer(dl, nhidden=2, depth=1, seq_length=seq_length)

nn = NeuralNetwork(arch, CPU(), Float64)

opt = Optimizer(AdamOptimizer(), nn)

batch = Batch(1000, seq_length)

loss_array = opt(nn, dl, batch, 1000, transformer_loss)