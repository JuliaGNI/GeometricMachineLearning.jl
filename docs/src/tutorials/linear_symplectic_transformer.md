# [Linear Symplectic Transformer](@id linear_symplectic_transformer_tutorial)

In this tutorial we compare the [linear symplectic transformer](@ref "Linear Symplectic Transformer") to the [standard transformer](@ref "Standard Transformer"). 

```@example lin_sympl_tran_tut
using GeometricMachineLearning # hide
using GeometricProblems.CoupledHarmonicOscillator: hodeensemble, default_parameters
using GeometricIntegrators: ImplicitMidpoint, integrate 
using LaTeXStrings
using Plots

const tstep = .3

const m₁ = default_parameters.m₁  
const m₂ = default_parameters.m₂ 
const k₁ = default_parameters.k₁ 
const k₂ = default_parameters.k₂ 
const k = [2.8, 3.5] 
 
params_collection = [(m₁ = m₁, m₂ = m₂, k₁ = k₁, k₂ = k₂, k = k_val) for k_val in k] 
# ensemble problem
ep = hodeensemble(; parameters = params_collection, tstep = tstep)

dl_nt = DataLoader(integrate(ep, ImplicitMidpoint()))
dl = DataLoader(vcat(dl_nt.input.q, dl_nt.input.p))

nothing # hide
```

We now define the architectures and train them: 

```@example lin_sympl_tran_tut
const seq_length = 2
const batch_size = 256
const n_epochs = 1000

arch_standard = StandardTransformerIntegrator(dl.input_dim; n_heads = 2)
arch_symplectic = LinearSymplecticTransformer(dl.input_dim, seq_length; n_sympnet = 2, L = 6, upscaling_dimension = 3 * dl.input_dim)
arch_sympnet = GSympNet(dl.input_dim; n_layers = 8, upscaling_dimension = 4 * dl.input_dim)

nn_standard = NeuralNetwork(arch_standard)
nn_symplectic = NeuralNetwork(arch_symplectic)
nn_sympnet = NeuralNetwork(arch_sympnet)

o_standard = Optimizer(AdamOptimizer(Float64), nn_standard)
o_symplectic = Optimizer(AdamOptimizer(Float64), nn_symplectic)
o_sympnet = Optimizer(AdamOptimizer(Float64), nn_sympnet)

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

We further try evaluating a trajectory with the trained networks: 

```@example lin_sympl_tran_tut
const index = 1
init_con = dl.input[:, 1:seq_length, index]

const n_steps = 50

function make_validation_plot(n_steps = n_steps)
    prediction_standard = iterate(nn_standard, init_con; n_points = n_steps, prediction_window = seq_length)
    prediction_symplectic = iterate(nn_symplectic, init_con; n_points = n_steps, prediction_window = seq_length)
    prediction_sympnet = iterate(nn_sympnet, init_con[:, 1]; n_points = n_steps)

    p_validate = plot(dl.input[1, 1:n_steps, index]; color = 1, ylabel = L"q_1", label = "implicit midpoint")
    plot!(p_validate, prediction_standard[1, :]; color = 2, label = "ST")
    plot!(p_validate, prediction_symplectic[1, :]; color = 4, label = "LST")
    plot!(p_validate, prediction_sympnet[1, :]; color = 3, label = "SympNet")

    p_validate
end

make_validation_plot()
```

We can see that the standard transformer is not able to stay close to the trajectory coming from implicit midpoint very well. If we plot the curves on an even longer time scale the fact that the standard transformer is non-symplectic will be even more pronounced: 

```@example lin_sympl_train_tut
make_validation_plot(500)
```