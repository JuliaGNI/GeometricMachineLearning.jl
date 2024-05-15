# [Linear Symplectic Transformer](@id linear_symplectic_transformer_tutorial)

In this tutorial we compare the [linear symplectic transformer](@ref "Linear Symplectic Transformer") to the [standard transformer](@ref "Standard Transformer"). 

```@example lin_sympl_tran_tut
using GeometricMachineLearning # hide
using GeometricProblems.CoupledHarmonicOscillator: hodeensemble, default_parameters
using GeometricIntegrators: ImplicitMidpoint, integrate 
using LaTeXStrings
using Plots
import Random

Random.seed!(123)

const tstep = .3
const n_init_con = 5

# ensemble problem
ep = hodeensemble([rand(2) for _ in 1:n_init_con], [rand(2) for _ in 1:n_init_con]; tstep = tstep)

dl_nt = DataLoader(integrate(ep, ImplicitMidpoint()))
dl = DataLoader(vcat(dl_nt.input.q, dl_nt.input.p))

nothing # hide
```

We now define the architectures and train them: 

```@example lin_sympl_tran_tut
const seq_length = 4
const batch_size = 16384
const n_epochs = 2000

arch_standard = StandardTransformerIntegrator(dl.input_dim; n_heads = 2, L = 1, n_blocks = 2)
arch_symplectic = LinearSymplecticTransformer(dl.input_dim, seq_length; n_sympnet = 2, L = 1, upscaling_dimension = 2 * dl.input_dim)
arch_sympnet = GSympNet(dl.input_dim; n_layers = 4, upscaling_dimension = 2 * dl.input_dim)

nn_standard = NeuralNetwork(arch_standard)
nn_symplectic = NeuralNetwork(arch_symplectic)
nn_sympnet = NeuralNetwork(arch_sympnet)

o_method = AdamOptimizerWithDecay(n_epochs, Float64)

o_standard = Optimizer(o_method, nn_standard)
o_symplectic = Optimizer(o_method, nn_symplectic)
o_sympnet = Optimizer(o_method, nn_sympnet)

batch = Batch(batch_size, seq_length)
batch2 = Batch(batch_size)

loss_array_standard = o_standard(nn_standard, dl, batch, n_epochs)
loss_array_symplectic = o_symplectic(nn_symplectic, dl, batch, n_epochs)
loss_array_sympnet = o_sympnet(nn_sympnet, dl, batch2, n_epochs)

p_train = plot(loss_array_standard; color = 2, xlabel = "epoch", ylabel = "training error", label = "ST", yaxis = :log)
plot!(p_train, loss_array_symplectic; color = 4, label = "LST")
plot!(p_train, loss_array_sympnet; color = 3, label = "SympNet")

p_train
```

We further evaluate a trajectory with the trained networks: 

```@example lin_sympl_tran_tut
const index = 1
init_con = dl.input[:, 1:seq_length, index]

const n_steps = 30

function make_validation_plot(n_steps = n_steps; kwargs...)
    prediction_standard = iterate(nn_standard, init_con; n_points = n_steps, prediction_window = seq_length)
    prediction_symplectic = iterate(nn_symplectic, init_con; n_points = n_steps, prediction_window = seq_length)
    prediction_sympnet = iterate(nn_sympnet, init_con[:, 1]; n_points = n_steps)

    p_validate = plot(dl.input[1, 1:n_steps, index]; color = 1, ylabel = L"q_1", label = "implicit midpoint", kwargs...)
    plot!(p_validate, prediction_standard[1, :]; color = 2, label = "ST", kwargs...)
    plot!(p_validate, prediction_symplectic[1, :]; color = 4, label = "LST", kwargs...)
    plot!(p_validate, prediction_sympnet[1, :]; color = 3, label = "SympNet", kwargs...)

    p_validate
end

make_validation_plot(; linewidth = 2)
```

We can see that the standard transformer is not able to stay close to the trajectory coming from implicit midpoint very well. The linear symplectic transformer outperforms the standard transformer as well as the SympNet while needed much fewer parameters than the standard transformer: 

```@example lin_sympl_tran_tut
parameterlength(nn_standard), parameterlength(nn_symplectic), parameterlength(nn_sympnet)
```

It is also interesting to note that the training error for the SympNet gets lower than the one for the linear symplectic transformer, but it does not manage to outperform it when looking at the validation. 