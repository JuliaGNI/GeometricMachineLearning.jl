# SympNet Documenation

Here is the documentation about the SympNets architecture that the package `GeometricMachineLearning.jl` offers. 

## Quick overview of the theory of SympNet

### Principle

SympNets is a new type of neural network proposing a new approach to compute the trajectory of a Hamiltonian system in phase space. Let us denote by $(q,p)\in \mathbb{R}^{2d}$ the phase space with $q\in \mathbb{R}^{d}$ the gereralized position and 
$p\in \mathbb{R}^{d}$ the generalized momentum. Given a physical problem, SympNets takes a phase space element $(q,p)$ and aims to compute the next position $(q',p')$ of the trajectory in phase space a time step later while preserving the well known symplectic structure of dynamical systems.
The way SympNet preserve the symplectic structure is really specific and characterizes it as this preseving is intrinsic of the neural network. Indeed, SympNet is not made with traditional layers but with symplectic layers (decribe later) modifyng the traditional universal approximation theorem into a symplectic one : SympNet is able to approach any symplectic function providing conditions on an activation function.

SympNet (noted $\Phi$ in the following) is so an integrator preserving symplecticity wich can compute, from an initial condition $(q_0,p_0)$, a sequence of phase space elements of a trajectory $(q_n,p_n)=\Phi(q_{n-1},p_{n-1})=...=\Phi^n(q_0,p_0)$. The time step between predictions is not a parameter we can choose but is related to the temporal frequency of the training data. SympNet can handle both  temporally regular data, i.e with a fix time step between data, and temporally irregular data, i.e with variable time step. 
 
### Architecture of SympNets

### Universal approximation theorems

## Development 
