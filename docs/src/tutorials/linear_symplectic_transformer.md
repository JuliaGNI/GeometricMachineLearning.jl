# [Linear Symplectic Transformer](@id linear_symplectic_transformer_tutorial)

In this tutorial we compare the [linear symplectic transformer](@ref "Linear Symplectic Transformer") to the [standard transformer](@ref tran). 

```@example lin_sympl_tran_tut
using GeometricMachineLearning # hide
using GeometricProblems.CoupledHarmonicOscillator: hodeensemble, default_parameters
using GeometricIntegrators: ImplicitMidpoint, integrate 

const m₁ = default_parameters.m₁  
const m₂ = default_parameters.m₂ 
const k₁ = default_parameters.k₁ 
const k₂ = default_parameters.k₂ 
const k = [0.0, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0] 
 
params_collection = [(m₁ = m₁, m₂ = m₂, k₁ = k₁, k₂ = k₂, k = k_val) for k_val in k] 
# ensemble problem
ep = hodeensemble(; parameters = params_collection)

dl_nt = DataLoader(integrate(ep, ImplicitMidpoint()))
dl = DataLoader(vcat(dl_nt.input.q, dl_nt.input.p))

nothing # hide
```

We now define the architectures and train them: 

```@example lin_sympl_tran_tut
const seq_length = 4
const batch_size = 32
const n_epochs = 10

arch_standard = StandardTransformerIntegrator(dl.input_dim)
arch_symplectic = LinearSymplecticTransformer(dl.input_dim, seq_length)

nn_standard = NeuralNetwork(arch_standard)
nn_symplectic = NeuralNetwork(arch_symplectic)

o_standard = Optimizer(AdamOptimizer(Float64), nn_standard)
o_symplectic = Optimizer(AdamOptimizer(Float64), nn_symplectic)

batch = Batch(batch_size, seq_length)

o_standard(nn_standard, dl, batch, n_epochs)
o_symplectic(nn_symplectic, dl, batch, n_epochs)
```