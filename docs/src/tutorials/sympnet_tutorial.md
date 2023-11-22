## SympNets with `GeometricMachineLearning.jl`

This page serves as a short introduction into using SympNets with `GeometricMachineLearning.jl`. For the general theory see [the theory section](../architectures/sympnet.md).

With `GeometricMachineLearning.jl`, it is really easy to implement and train a SympNet. The steps are the following :
- __Create the architecture__ in one line with the function `GSympNet` or `LASympNet`,
- __Create the neural networks__ depending a backend (e.g. with Lux),
- __Create an optimizer__ for the training step,
- __Train__ the neural networks with the `train!`function.

Both LA-SympNet and G-SympNet architectures can be generated in one line with `GeometricMachineLearning.jl`.

### LA-SympNet

To create a LA-SympNet, one needs to write

```julia
lasympnet = LASympNet(dim; width=5, nhidden=1, activation=tanh, init_uplow_linear=[true,false], 
            init_uplow_act=[true,false],init_sym_matrices=Lux.glorot_uniform, init_bias=Lux.zeros32, 
            init_weight=Lux.glorot_uniform) 
```
`LASympNet` takes one obligatory argument:
- __dim__ : the dimension of the phase space,

and several keywords argument :
- __width__ : the width for all the symplectic linear layers with default value set to 5 (if width>5, width is set to 5),
- __nhidden__ : the number of pairs of symplectic linear and activation layers with default value set to 0 (i.e LA-SympNet is a single symplectic linear layer),
- __activation__ : the activation function for all the symplectic activations layers with default value set to tanh,
- __init_uplow_linear__ : a vector of boolean whose the ith coordinate is true only if all the symplectic linear layers in (i mod `length(init_uplow_linear)`)-th position is up (for example the default value is [true,false] which represents an alternation of up and low symplectic linear layers),
- __init_uplow_act__ : a vector of boolean whose the ith coordinate is true only if all the symplectic activation layers in (i mod `length(init_uplow_act)`)-th position is up (for example the default value is [true,false] which represents an alternation of up and low symplectic activation layers),
- __init_sym_matrices__: the function which gives the way to initialize the symmetric matrices $S^i$ of symplectic linear layers,
- __init_bias__: the function which gives the way to initialize the vector of bias $b$,
- __init_weight__: the function which gives the way to initialize the weight $a$.

The default value of the last three keyword arguments uses Lux functions.

### G-SympNet

 To create a G-SympNet, one needs to write

```julia
gsympnet = GSympNet(dim; width=dim, nhidden=1, activation=tanh, init_uplow=[true,false], init_weight=Lux.glorot_uniform, 
init_bias=Lux.zeros32, init_scale=Lux.glorot_uniform) 
```
`GSympNet` takes one obligatory argument:
- __dim__ : the dimension of the phase space,

and severals keywords argument :
- __width__ : the width for all the gradients layers with default value set to dim to have width$\geq$dim,
- __nhidden__ : the number of gradient layers with default value set to 1,
- __activation__ : the activation function for all the gradients layers with default value set to tanh,
- __init_uplow__: a vector of boolean whose the ith coordinate is true only if all the gradient layers in (i mod `length(init_uplow)`)-th position is up (for example the default value is [true,false] which represents an alternation of up and low gradient layers),
- __init_weight__: the function which gives the way to initialize the vector of weights $a$,
- __init_bias__: the function which gives the way to initialize the vector of bias $b$,
- __init_scale__: the function which gives the way to initialize the scale matrix $K$.

The default value of the last three keyword arguments uses Lux functions.

### Loss function

To train the SympNet, one need data along a trajectory such that the model is trained to perform an integration. These data are $(Q,P)$ where $Q[i,j]$ (respectively $P[i,j]$) is the real number $q_j(t_i)$ (respectively $p[i,j]$) which is the j-th coordinates of the generalized position (respectively momentum) at the i-th time step. One also need a loss function defined as :

$$Loss(Q,P) = \underset{i}{\sum} d(\Phi(Q[i,-],P[i,-]), [Q[i,-] P[i,-]]^T)$$
where $d$ is a distance on $\mathbb{R}^d$.

## Data Structures in `GeometricMachineLearning.jl`

![](../tikz/structs_visualization.png)

## Examples

Let us see how to use it on several examples.

### Example of a pendulum with G-SympNet

Let us begin with a simple example, the pendulum system, the Hamiltonian of which is 
```math
H:(q,p)\in\mathbb{R}^2 \mapsto \frac{1}{2}p^2-cos(q) \in \mathbb{R}.
```

Here we generate pendulum data with the script `GeometricMachineLearning/scripts/pendulum.jl`:

```@example sympnet
using GeometricMachineLearning

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
const upscaling_dimension = 10
# hidden layers
const nhidden = 1
# activation function
const activation = tanh

# calling G-SympNet architecture 
gsympnet = GSympNet(dl, upscaling_dimension=upscaling_dimension, nhidden=nhidden, activation=activation)

# calling LA-SympNet architecture 
lasympnet = LASympNet(dl, nhidden=nhidden, activation=activation)

# specify the backend
backend = CPU()

# initialize the networks
la_nn = NeuralNetwork(lasympnet, backend, type) 
g_nn = NeuralNetwork(gsympnet, backend, type)
nothing # hide
```

*Remark*: We can also specify whether we would like to start with a layer that changes the $q$-component or one that changes the $p$-component. This can be done via the keywords `init_upper` for `GSympNet`, and `init_upper_linear` and `init_upper_act` for `LASympNet`.

We have to define an optimizer which will be use in the training of the SympNet. For more details on optimizer, please see the [corresponding documentation](../Optimizer.md). In this example we use [Adam](../optimizers/adam_optimizer.md):

```@example sympnet
# set up optimizer; for this we first need to specify the optimization method (argue for why we need the optimizer method)
opt_method = AdamOptimizer(; T=type)
la_opt = Optimizer(opt_method, la_nn)
g_opt = Optimizer(opt_method, g_nn)
nothing # hide
```

We can now perform the training of the neural networks. The syntax is the following :

```@example sympnet
# number of training epochs
const nepochs = 1000
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
The train function will change the parameters of the neural networks and gives an a vector containing the evolution of the value of the loss function during the training. Default values for the arguments `ntraining` and `batch_size` are respectively $1000$ and $10$.

The trainings data `data_q` and `data_p` must be matrices of $\mathbb{R}^{n\times d}$ where $n$ is the length of data and $d$ is the half of the dimension of the system, i.e `data_q[i,j]` is $q_j(t_i)$ where $(t_1,...,t_n)$ are the corresponding time of the training data.

Then we can make prediction. Let's compare the initial data with a prediction starting from the same phase space point using the provided function Iterate_Sympnet:

```@example sympnet
ics = (q=qp_data.q[:,1], p=qp_data.p[:,1])

steps_to_plot = 200

#predictions
la_trajectory = Iterate_Sympnet(la_nn, ics; n_points = steps_to_plot)
g_trajectory = Iterate_Sympnet(g_nn, ics; n_points = steps_to_plot)

using Plots
p2 = plot(qp_data.q'[1:steps_to_plot], qp_data.p'[1:steps_to_plot], label="training data")
plot!(p2, la_trajectory.q', la_trajectory.p', label="LA Sympnet")
plot!(p2, g_trajectory.q', g_trajectory.p', label="G Sympnet")
```

We see that `GSympNet` gives an almost perfect math on the training data whereas `LASympNet` cannot even properly replicate the training data. It also takes longer to train `LASympNet`.