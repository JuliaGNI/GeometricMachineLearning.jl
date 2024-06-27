# SympNets with `GeometricMachineLearning`

This page serves as a short introduction into using SympNets with `GeometricMachineLearning.jl`. For the general theory see [the theory section](../architectures/sympnet.md).

With `GeometricMachineLearning.jl` one can easily implement SympNets. The steps are the following :
- __Specify the architecture__ with the functions `GSympNet` and `LASympNet`,
- __Specify the type and the backend__ with `NeuralNetwork`,
- __Pick an optimizer__ for training the network,
- __Train__ the neural networks!

We discuss these points is some detail:

## Specifying the architecture

To call an $LA$-SympNet, one needs to write

```julia
lasympnet = LASympNet(dim; depth=5, nhidden=1, activation=tanh, init_upper_linear=true, init_upper_act=true) 
```
`LASympNet` takes one obligatory argument:
- __dim__ : the dimension of the phase space (i.e. an integer) or optionally an instance of `DataLoader`. This latter option will be used below.

and several keywords argument :
- __depth__ : the depth for all the linear layers. The default value set to 5 (if width>5, width is set to 5). See the [theory section](../architectures/sympnet.md) for more details; there **depth** was called $n$.
- __nhidden__ : the number of pairs of linear and activation layers with default value set to 1 (i.e the $LA$-SympNet is a composition of a linear layer, an activation layer and then again a single layer). 
- __activation__ : the activation function for all the activations layers with default set to tanh,
- __init_upper_linear__ : a boolean that indicates whether the first linear layer changes $q$ first. By default this is `true`.
- __init_upper_act__ : a boolean that indicates whether the first activation layer changes $q$ first. By default this is `true`.

### G-SympNet

 To call a G-SympNet, one needs to write

```julia
gsympnet = GSympNet(dim; upscaling_dimension=2*dim, n_layers=2, activation=tanh, init_upper=true) 
```
`GSympNet` takes one obligatory argument:
- __dim__ : the dimension of the phase space (i.e. an integer) or optionally an instance of `DataLoader`. This latter option will be used below.

and severals keywords argument :
- __upscaling_dimension__: The first dimension of the matrix with which the input is multiplied. In the [theory section](../architectures/sympnet.md) this matrix is called $K$ and the *upscaling dimension* is called $m$.
- __n_layers__: the number of gradient layers with default value set to 2.
- __activation__ : the activation function for all the activations layers with default set to tanh.
- __init_upper__ : a boolean that indicates whether the first gradient layer changes $q$ first. By default this is `true`.

### Loss function

The loss function described in the [theory section](../architectures/sympnet.md) is the default choice used in `GeometricMachineLearning.jl` for training SympNets.

## Examples

Let us see how to use it on several examples.

### Example of a pendulum with G-SympNet

Let us begin with a simple example, the pendulum system, the Hamiltonian of which is 
```math
H:(q,p)\in\mathbb{R}^2 \mapsto \frac{1}{2}p^2-cos(q) \in \mathbb{R}.
```

Here we generate pendulum data with the script `GeometricMachineLearning/scripts/pendulum.jl`:

```@example sympnet
using GeometricMachineLearning # hide
import Random # hide

Random.seed!(1234)

# load script
include("../../../scripts/pendulum.jl")
# specify the data type
type = Float16 
# get data 
qp_data = GeometricMachineLearning.apply_toNT(a -> type.(a), pendulum_data((q=[0.], p=[1.]); tspan=(0.,100.)))
# call the DataLoader
dl = DataLoader(qp_data)
# this last line is a hack so as to not display the output # hide
nothing # hide
```

Next we specify the architectures. `GeometricMachineLearning.jl` provides useful defaults for all parameters although they can be specified manually (which is done in the following):

```@example sympnet
# layer dimension for gradient module 
const upscaling_dimension = 2
# hidden layers
const nhidden = 1
# activation function
const activation = tanh

# calling G-SympNet architecture 
gsympnet = GSympNet(dl, upscaling_dimension=upscaling_dimension, n_layers=4, activation=activation)

# calling LA-SympNet architecture 
lasympnet = LASympNet(dl, nhidden=nhidden, activation=activation)

# specify the backend
const backend = CPU()

# initialize the networks
la_nn = NeuralNetwork(lasympnet, backend, type) 
g_nn = NeuralNetwork(gsympnet, backend, type)
nothing # hide
```

If we want to obtain information on the number of parameters in a neural network, we can do that very simply with the function `parameterlength`. For the `LASympNet`:
```@example sympnet
parameterlength(la_nn.model)
```

And for the `GSympNet`:
```@example sympnet
parameterlength(g_nn.model)
```

*Remark*: We can also specify whether we would like to start with a layer that changes the $q$-component or one that changes the $p$-component. This can be done via the keywords `init_upper` for `GSympNet`, and `init_upper_linear` and `init_upper_act` for `LASympNet`.

We have to define an optimizer which will be use in the training of the SympNet. For more details on optimizer, please see the [corresponding documentation](@ref "Neural Network Optimizers"). In this example we use [Adam](@ref "The Adam Optimizer"):

```@example sympnet
# set up optimizer; for this we first need to specify the optimization method (argue for why we need the optimizer method)
opt_method = AdamOptimizer(type)
la_opt = Optimizer(opt_method, la_nn)
g_opt = Optimizer(opt_method, g_nn)
nothing # hide
```

We can now perform the training of the neural networks. The syntax is the following :

```@example sympnet
# number of training epochs
const nepochs = 300
# Batchsize used to compute the gradient of the loss function with respect to the parameters of the neural networks.
const batch_size = 100

batch = Batch(batch_size)

# perform training (returns array that contains the total loss for each training step)
g_loss_array = g_opt(g_nn, dl, batch, nepochs)
la_loss_array = la_opt(la_nn, dl, batch, nepochs)
nothing # hide
```

We can also plot the training errors against the epoch (here the $y$-axis is in log-scale):
```@example sympnet
using Plots
p1 = plot(g_loss_array, xlabel="Epoch", ylabel="Training error", label="G-SympNet", color=3, yaxis=:log)
plot!(p1, la_loss_array, label="LA-SympNet", color=2)
```

The trainings data `data_q` and `data_p` must be matrices of $\mathbb{R}^{n\times d}$ where $n$ is the length of data and $d$ is the half of the dimension of the system, i.e `data_q[i,j]` is $q_j(t_i)$ where $(t_1,...,t_n)$ are the corresponding time of the training data.

Now we can make a prediction. Let's compare the initial data with a prediction starting from the same phase space point using the function `iterate`:

```@example sympnet
ics = (q=qp_data.q[:,1], p=qp_data.p[:,1])

steps_to_plot = 200

#predictions
la_trajectory = iterate(la_nn, ics; n_points = steps_to_plot)
g_trajectory =  iterate(g_nn, ics; n_points = steps_to_plot)

using Plots
p2 = plot(qp_data.q'[1:steps_to_plot], qp_data.p'[1:steps_to_plot], label="training data")
plot!(p2, la_trajectory.q', la_trajectory.p', label="LA Sympnet")
plot!(p2, g_trajectory.q', g_trajectory.p', label="G Sympnet")
```

We see that `GSympNet` outperforms the `LASympNet` on this problem.