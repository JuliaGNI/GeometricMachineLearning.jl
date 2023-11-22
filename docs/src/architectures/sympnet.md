# SympNet

This document discusses the SympNet architecture and its implementation in `GeometricMachineLearning.jl`.

## Quick overview of the theory of SympNet

### Principle

SympNets (see [jin2020sympnets](@cite) for the eponymous paper) are a type of neural network proposing a new approach to compute the trajectory of a Hamiltonian system in phase space. Take $(q,p)=(q_1,\ldots,q_d,p_1,\ldots,p_d)^T\in \mathbb{R}^{2d}$ as the coordinates in phase space, where $q=(q_1, \ldots, q_d)\in \mathbb{R}^{d}$ is refered to as the *position* and $p=(p_1, \ldots, p_d)\in \mathbb{R}^{d}$ the *momentum*. Given a point $(q,p)$ in $\mathbb{R}^{2d}$ the SympNet aims to compute the *next position* $(q',p')$ and thus predicts the trajectory while preserving the *symplectic structure* of the system.
SympNets are enforcing symplecticity strongly, meaning that this property is hard-coded into the network architecture. The layers are reminiscent of traditional neural network feedforward layers, but have a strong restriction imposed on them in order to be symplectic.

SympNets (denoted by $\Phi$ in the following) can be viewed as a ``symplectic integrator'' (see [hairer2006geometric](@cite) and [leimkuhler2004simulating](@cite)). Its goal is to predict, based on an initial condition $((q^{(0)})^T,(p^{(0)})^T)^T$, a sequence of points in phase space $((q^{(0)})^T,(p^{(0)})^T)^T, ((q^{(1)})^T,(p^{(1)})^T)^T, \ldots, ((q^{(n)})^T,(p^{(n)})^T)^T$ that fit the training data as well as possible. The time step between predictions is not a parameter we can choose but is *related to the temporal frequency of the training data*. SympNet can handle both  temporally regular data, i.e with a fix time step between data, and temporally irregular data, i.e with variable time step. 

### Architecture of SympNets

With `GeometricMachineLearning.jl`, it is possible to implement two types of SympNet architectures: $LA$-SympNets and $G$-SympNets. 
 
#### LA-SympNet

![](../tikz/sympnet_architecture.png)

$LA$-SympNets are made of the alternation of two types of layers, symplectic linear layers and symplectic activation layers.  For a given integer $n$, a symplectic linear layer is defined by

```math
\mathcal{L}^{n,up}
\begin{pmatrix}
 q \\
 p \\
\end{pmatrix}
 =  
\begin{pmatrix} 
 I & S^n/0 \\
 0/S^n & I \\
\end{pmatrix}
 \cdots 
\begin{pmatrix} 
 I & 0 \\
 S^2 & I \\
\end{pmatrix}
\begin{pmatrix} 
 I & S^1 \\
 0 & I \\
\end{pmatrix}
\begin{pmatrix}
 q \\
 p \\
\end{pmatrix}
+ b ,
```
 
or 
 
```math
\mathcal{L}^{n,low}
\begin{pmatrix}  q  \\  
 p  \end{pmatrix} =  
  \begin{pmatrix} 
 I & 0/S^n  \\ 
 S^n/0 & I
 \end{pmatrix} \cdots 
  \begin{pmatrix} 
 I & S^2  \\ 
 0 & I
 \end{pmatrix}
 \begin{pmatrix} 
 I & 0  \\ 
 S^1 & I
 \end{pmatrix}
 \begin{pmatrix}  q  \\  
 p  \end{pmatrix}
  + b . 
```

The learnable parameters are the symmetric matrices $S^i\in\mathbb{R}^{d\times d}$ and the bias $b\in\mathbb{R}^{2d}$. The integer $n$ is the width of the symplectic linear layer. If $n\geq5$, we know that the symplectic linear layers represent any linear symplectic map so that $n$ need not be larger than 5 (see [jin2022optimal](@cite)). We note the set of symplectic linear layers $\mathcal{M}^L$. This type of layers plays the role of standard linear layers. 

For a given activation function $\sigma$, a symplectic activation layer is defined by

```math
 \mathcal{A}^{up}  \begin{pmatrix}  q  \\  
 p  \end{pmatrix} =  
  \begin{bmatrix} 
 I&\hat{\sigma}^{a}  \\ 
 0&I
 \end{bmatrix} \begin{pmatrix}  q  \\  
 p  \end{pmatrix} :=
 \begin{pmatrix} 
  \mathrm{diag}(a)\sigma(p)+q \\ 
  p
 \end{pmatrix},
```
 
 or
 
```math
 \mathcal{A}^{low}  \begin{pmatrix}  q  \\  
 p  \end{pmatrix} =  
  \begin{bmatrix} 
 I&0  \\ 
 \hat{\sigma}^{a}&I
 \end{bmatrix} \begin{pmatrix}  q  \\  
 p  \end{pmatrix}
 :=
 \begin{pmatrix} 
 q \\ 
 \mathrm{diag}(a)\sigma(q)+p
 \end{pmatrix}.
```
 
The *scaling vector* $a\in\mathbb{R^{d}}$ constitutes the learnable weights. This type of layer plays the role of a standard activation layer. We denote the set of symplectic activation layers by $\mathcal{M}^A$. 
 
A $LA$-SympNet is a function of the form $\Psi=l_{k+1} \circ a_{k} \circ v_{k} \circ \cdots \circ a_1 \circ l_1$ where $(l_i)_{1\leq i\leq k+1} \subset (\mathcal{M}^L)^{k+1}$ and $(a_i)_{1\leq i\leq k} \subset (\mathcal{M}^A)^{k}$.
 
 #### $G$-SympNets
 
 $G$-SympNets are an alternative to $LA$-SympNets. They are built with only one kind of layer, called *gradient layer*. For a given activation function $\sigma$ and an integer $n\geq d$, a gradient layers is a symplectic map from $\mathbb{R}^{2d}$ to $\mathbb{R}^{2d}$ defined by
 
```math
 \mathcal{G}^{up}  \begin{pmatrix}  q  \\  
 p  \end{pmatrix} =  
  \begin{bmatrix} 
 I&\hat{\sigma}^{K,a,b}  \\ 
 0&I
 \end{bmatrix} \begin{pmatrix}  q  \\  
 p  \end{pmatrix} :=
 \begin{pmatrix} 
  K^T \mathrm{diag}(a)\sigma(Kp+b)+q \\ 
  p
 \end{pmatrix},
```
 
or
 
```math
 \mathcal{G}^{low}  \begin{pmatrix}  q  \\  
 p  \end{pmatrix} =  
  \begin{bmatrix} 
 I&0  \\ 
 \hat{\sigma}^{K,a,b}&I
 \end{bmatrix} \begin{pmatrix}  q  \\  
 p  \end{pmatrix}
 :=
 \begin{pmatrix} 
 q \\ 
 K^T \mathrm{diag}(a)\sigma(Kq+b)+p
 \end{pmatrix}.
```

Note here the different roles played by round and square braces, the latter indicates a nonlinear operation as opposed to a regular vector or matrix. The parameters of this layer are the *scaling matrix* $K\in\mathbb{R}^{n\times d}$, the bias $b\in\mathbb{R}^{n}$ and the *scaling vector* $a\in\mathbb{R}^{n}$. The name ``gradient layer'' has its origin in the fact that the expression $[K^T\mathrm{diag}(a)\sigma(Kq+b)]_i = \sum_jk_{ji}a_j\sigma(\sum_\ell{}k_{j\ell}q_\ell+b_j)$ is the gradient of a function $\sum_ja_j\tilde{\sigma}(\sum_\ell{}k_{j\ell}q_\ell+b_j)$, where $\tilde{\sigma}$ is the antiderivative of $\sigma$.
 
If we denote by $\mathcal{M}^G$ the set of gradient layers, a $G$-SympNet is a function of the form $\Psi=g_k \circ g_{k-1} \circ \cdots \circ g_1$ where $(g_i)_{1\leq i\leq k} \subset (\mathcal{M}^G)^k$.

### Universal approximation theorems

In order to state the \textit{universal approximation theorem} for both architectures we first need a few definitions:
 
Let $U$ be an open set of $\mathbb{R}^{2d}$, and let us denote by $\mathcal{SP}^r(U)$ the set of $C^r$ smooth symplectic maps on $U$. We now define a topology on $C^r(K, \mathbb{R}^n)$, the set of $C^r$-smooth maps from a compact set $K\subset\mathbb{R}^{n}$ to $\mathbb{R}^{n}$ through the norm

```math
||f||_{C^r(K,\mathbb{R}^{n})} = \underset{|\alpha|\leq r}{\sum} \underset{1\leq i \leq n}{\max}\underset{x\in K}{\sup} |D^\alpha f_i(x)|,
```
where the differential operator $D^\alpha$ is defined by 
```math
D^\alpha f = \frac{\partial^{|\alpha|} f}{\partial x_1^{\alpha_1}...x_n^{\alpha_n}},
```
with $|\alpha| = \alpha_1 +...+ \alpha_n$. 

__Definition__ $\sigma$ is **$r$-finite** if $\sigma\in C^r(\mathbb{R},\mathbb{R})$ and $\int |D^r\sigma(x)|dx <+\infty$.


__Definition__ Let $m,n,r\in \mathbb{N}$ with $m,n>0$ be given, $U$ an open set of $\mathbb{R}^m$, and $I,J\subset C^r(U,\mathbb{R}^n)$. We say $J$ is **$r$-uniformly dense on compacta in $I$** if $J \subset I$ and for any $f\in I$, $\epsilon>0$, and any compact $K\subset U$, there exists $g\in J$ such that $||f-g||_{C^r(K,\mathbb{R}^{n})} < \epsilon$.

We can now state the universal approximation theorems:

__Theorem (Approximation theorem for LA-SympNet)__ For any positive integer $r>0$ and open set $U\in \mathbb{R}^{2d}$, the set of $LA$-SympNet is $r$-uniformly dense on compacta in $SP^r(U)$ if the activation function $\sigma$ is $r$-finite.

__Theorem (Approximation theorem for G-SympNet)__ For any positive integer $r>0$ and open set $U\in \mathbb{R}^{2d}$, the set of $G$-SympNet is $r$-uniformly dense on compacta in $SP^r(U)$ if the activation function $\sigma$ is $r$-finite.

There are many $r$-finite activation functions commonly used in neural networks, for example:
- sigmoid $\sigma(x)=\frac{1}{1+e^{-x}}$ for any positive integer $r$, 
- tanh $\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$ for any positive integer $r$. 


## SympNet with `GeometricMachineLearning.jl`


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

```@example
using GeometricMachineLearning

# load script
include("../../scripts/pendulum.jl")
# specify the data type
type = Float16 
# get data 
qp_data = GeometricMachineLearning.apply_toNT(a -> type.(a), pendulum_data((q=[0.], p=[1.]); tspan=(0.,100.)))
# call the DataLoader
dl = DataLoader(qp_data)
```

Next we specify the architectures. `GeometricMachineLearning.jl` provides useful defaults for all parameters although they can be specified manually (which is done in the following):

```@example
# layer dimension for gradient module 
const upscaling_dimension = 10
# hidden layers
const nhidden = 2
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
```

*Remark*: We can also specify whether we would like to start with a layer that changes the $q$-component or one that changes the $p$-component. This can be done via the keywords `init_upper` for `GSympNet`, and `init_upper_linear` and `init_upper_act` for `LASympNet`.

We have to define an optimizer which will be use in the training of the SympNet. For more details on optimizer, please see the [corresponding documentation](../Optimizer.md). In this example we use [Adam](../optimizers/adam_optimizer.md):

```@example
# set up optimizer; for this we first need to specify the optimization method (argue for why we need the optimizer method)
opt_method = AdamOptimizer(; T=type)
la_opt = Optimizer(opt_method, la_nn)
g_opt = Optimizer(opt_method, g_nn)
```

We can now perform the training of the neural networks. The syntax is the following :

```@example
# number of training epochs
const nepochs = 1000
# Batchsize used to compute the gradient of the loss function with respect to the parameters of the neural networks.
const batch_size = 100

batch = Batch(batch_size)

# perform training (returns array that contains the total loss for each training step)
g_loss_array = g_opt(g_nn, dl, batch, nepochs)
la_loss_array = la_opt(la_nn, dl, batch, nepochs)
```
The train function will change the parameters of the neural networks and gives an a vector containing the evolution of the value of the loss function during the training. Default values for the arguments `ntraining` and `batch_size` are respectively $1000$ and $10$.

The trainings data `data_q` and `data_p` must be matrices of $\mathbb{R}^{n\times d}$ where $n$ is the length of data and $d$ is the half of the dimension of the system, i.e `data_q[i,j]` is $q_j(t_i)$ where $(t_1,...,t_n)$ are the corresponding time of the training data.

Then we can make prediction. Let's compare the initial data with a prediction starting from the same phase space point using the provided function Iterate_Sympnet:

```@example
ics = (q=qp_data.q[:,1], p=qp_data.p[:,1])

steps_to_plot = 200

#predictions
la_trajectory = Iterate_Sympnet(la_nn, ics; n_points = steps_to_plot)
g_trajectory = Iterate_Sympnet(g_nn, ics; n_points = steps_to_plot)

using Plots
p = plot(qp_data.q'[1:steps_to_plot], qp_data.p'[1:steps_to_plot], label="training data")
plot!(p, la_trajectory.q', la_trajectory.p', label="LA Sympnet")
plot!(p, g_trajectory.q', g_trajectory.p', label="G Sympnet")
```

We see that `GSympNet` gives an almost perfect math on the training data whereas `LASympNet` cannot even properly replicate the training data. It also takes longer to train `LASympNet`.