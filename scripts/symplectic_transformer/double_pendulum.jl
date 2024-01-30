using GeometricMachineLearning: DataLoader, LinearSymplecticTransformer, NeuralNetwork, CPU, Batch, AdamOptimizer, Optimizer, transformer_loss, GSympNet
using GeometricProblems.DoublePendulum: tspan, tstep, default_parameters, hodeproblem
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricEquations: EnsembleProblem
using CUDA

initial_conditions = [
    (q = [π / i, π / j], p = [0.0, π / k]) for i=1:10, j=1:10, k=1:5
]
initial_conditions = reshape(initial_conditions, length(initial_conditions))

ensemble_problem = EnsembleProblem(hodeproblem().equation, (tspan[1], 20*tspan[2]), tstep, initial_conditions, default_parameters)

ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())

dl = DataLoader(ensemble_solution)

const seq_length = 5
const nhidden = 4
const depth = 1 
const backend = CUDABackend()
const T = backend == CUDABackend() ? Float32 : Float64 
const batch_size = 1000
const n_epochs = 100
const opt_method = AdamOptimizer(T)

dl = backend == CUDABackend() ? DataLoader(dl.input |> cu) : dl

function train_linear_symplectic_transformer()

    arch = LinearSymplecticTransformer(dl, nhidden = nhidden, depth = depth, seq_length = seq_length)

    nn = NeuralNetwork(arch, backend, T)

    opt = Optimizer(opt_method, nn)

    batch = Batch(batch_size, seq_length)

    loss_array = opt(nn, dl, batch, n_epochs, transformer_loss)

    loss_array, nn
end 

function train_sympnet()

    arch = GSympNet(dl, nhidden = nhidden * depth)

    nn = NeuralNetwork(arch, backend, T)

    opt = Optimizer(opt_method, nn)

    batch = Batch(batch_size, 1)

    loss_array = opt(nn, dl, batch, n_epochs)

    loss_array, nn
end

sympnet_loss_array, sympnet = train_sympnet()
transformer_loss_array, transformer = train_linear_symplectic_transformer()
