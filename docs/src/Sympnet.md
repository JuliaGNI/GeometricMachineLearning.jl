# SympNet Documenation

Here is the documentation about the SympNets architecture that the package `GeometricMachineLearning.jl` offers. 

## Quick overview of the theory of SympNet

### Principle

SympNets is a new type of neural network proposing a new approach to compute the trajectory of an Hamiltonian system in phase space. Let us denote by $(q,p)\in \mathbb{R}^{2d}$ the phase space with $q\in \mathbb{R}^{d}$ the gereralized position and 
$p\in \mathbb{R}^{d}$ the generalized momentum. Given a physical problem, SympNets takes a phase space element $(q,p)$ and aims to compute the next position $(q',p')$ of the trajectory in phase space a time step later while preserving the well known symplectic structure of Hamiltonian systems.
The way SympNet preserve the symplectic structure is really specific and characterizes it as this preseving is intrinsic of the neural network. Indeed, SympNet is not made with traditional layers but with symplectic layers (decribe later) modifyng the traditional universal approximation theorem into a symplectic one : SympNet is able to approach any symplectic function providing conditions on an activation function.

SympNet (noted $\Phi$ in the following) is so an integrator preserving symplecticity wich can compute, from an initial condition $(q_0,p_0)$, a sequence of phase space elements of a trajectory $(q_n,p_n)=\Phi(q_{n-1},p_{n-1})=...=\Phi^n(q_0,p_0)$. The time step between predictions is not a parameter we can choose but is related to the temporal frequency of the training data. SympNet can handle both  temporally regular data, i.e with a fix time step between data, and temporally irregular data, i.e with variable time step. 
 
### Architecture of SympNets

### Universal approximation theorems

## SympNet with `GeometricMachineLearning.jl`

With `GeometricMachineLearning.jl`, it is really easy to implement and train a SympNet. Let us see how to use it on severals examples.

### Example of a pendulum

 The first thing to do is to create an architecture, either a LA-SympNet of a G-SympNet. Both needs the dimension of the system (2 in the case of a pendulum) and takes two optional parameters : the numer of hidden layer, and an activation function, with respective default values $1$ and $\tanh$.
 G-SympNet haves an additional optional parameters with default value sets to the dimension of the system which is the size of the gradient layers.
 
```julia
# number of inputs/dimension of system
const ninput = 2
# layer dimension for gradient module 
const ld = 10 
# hidden layers
const ln = 2
# activation function
const act = tanh

# Creation of a G-SympNet architecture 
gsympnet = GSympNet(ninput, width=ld, nhidden=ln, activation=act)

# Creation of a LA-SympNet architecture 
lasympnet = LASympNet(ninput, nhidden=ln, activation=act)
```
 We will follow the example with a G-SympNet but it's exactly the same for LA-SympNets. Then we can create the neraul networks depending on the backend. Here we will use Lux :

```julia
# create Lux network
nn = NeuralNetwork(gsympnet, LuxBackend())
```

We have to define an optimizer wich will be use in the training of the SympNet. For more details on optimizer, please see the corresponding documentation [Optimizer.jl](./Optimizer.md)
