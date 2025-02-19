# SympNet Architecture

This section discusses the symplectic neural network (SympNet) architecture and its implementation in `GeometricMachineLearning`.

## Principle

SympNets [jin2020sympnets](@cite) are a type of neural network that can model the trajectory of a [canonical Hamiltonian system](@ref "Symplectic Systems") in phase space. Take ``(q^T,p^T)^T=(q_1,\ldots,q_d,p_1,\ldots,p_d)^T\in \mathbb{R}^{2d}`` as the coordinates in phase space, where ``q=(q_1, \ldots, q_d)^T\in \mathbb{R}^{d}`` is refered to as the *position* and ``p=(p_1, \ldots, p_d)^T\in \mathbb{R}^{d}`` the *momentum*. Given a point 
```math
  \begin{pmatrix} q^{(m)} \\ p^{(m)} \end{pmatrix} \in \mathbb{R}^{2d}
```
the SympNet aims to compute the *next position*
```math
  \mathrm{SympNet}\left( \begin{pmatrix} q^{(m)} \\ p^{(m)} \end{pmatrix} \right) = \begin{pmatrix} \tilde{q}^{(m+1)} \\ \tilde{p}^{(m+1)} \end{pmatrix} \in \mathbb{R}^{2d}
``` 
and thus predicts the trajectory while preserving the *symplectic structure* of the system.
SympNets are enforcing symplecticity strongly, meaning that this property is hard-coded into the network architecture. The layers are reminiscent of traditional neural network feedforward layers, but have a strong restriction imposed on them in order to be symplectic.

SympNets can be viewed as a *symplectic integrator* or symplectic one-step method[^1] [hairer2006geometric, leimkuhler2004simulating](@cite). Their goal is to predict, based on an initial condition ``((q^{(0)})^T,(p^{(0)})^T)^T``, a sequence of points in phase space that fit the training data as well as possible:

[^1]: *Symplectic multi-step methods* can be modeled with [transformers](@ref "Linear Symplectic Transformer").

```math
\begin{pmatrix} q^{(0)} \\ p^{(0)} \end{pmatrix}, \begin{pmatrix} \tilde{q}^{(1)} \\ \tilde{p}^{(1)} \end{pmatrix}, \cdots \begin{pmatrix} \tilde{q}^{(n)} \\ \tilde{p}^{(n)} \end{pmatrix}.
```
The tilde in the above equation indicates *predicted data*. With standard SympNets[^2] the time step between predictions is not a parameter we can choose but is related to the *temporal frequency of the training data*. This means that if data is recorded in an interval of e.g. 0.1 seconds, then this will be the time step of our integrator.

[^2]: Recently an approach [horn4555181generalized](@cite) has been proposed that makes explicitly specifying the time step possible by viewing SympNets as a subclass of so-called "Generalized Hamiltonian Neural Networks".

SympNets preserve symplecticity by exploiting the ``(q, p)`` structure of the system. This is visualized below:

![Visualization of the SympNet architecture.](../tikz/sympnet_architecture_light.png)
![Visualization of the SympNet architecture.](../tikz/sympnet_architecture_dark.png)

In the figure above we see that an update for ``q`` is based on data coming from ``p`` and an update for ``p`` is based on data coming from ``q``. ``T_i:\mathbb{R}^d\to\mathbb{R}^d`` is an operation that changes ``p`` when ``i`` is even and changes ``q`` when odd. It has the special property that its Jacobian is a symmetric matrix. There are two types of SympNet architectures: ``LA``-SympNets and ``G``-SympNets. 
 
## ``LA``-SympNet

The first type of SympNets, ``LA``-SympNets, are obtained from composing two types of layers: symplectic linear layers and [symplectic activation layers](@ref "SympNet Gradient Layer").  For a given integer ``w``, the *linear part* of an ``LA``-SympNet is

```math
\mathcal{L}^{w,\mathrm{up}}
\begin{pmatrix}
 q \\
 p \\
\end{pmatrix}
 =  
\begin{pmatrix} 
 \mathbb{I} & A^w/\mathbb{O} \\
 \mathbb{O}/A^w & \mathbb{I} \\
\end{pmatrix}
 \cdots 
\begin{pmatrix} 
 \mathbb{I} & \mathbb{O} \\
 A^2 & \mathbb{I} \\
\end{pmatrix}
\begin{pmatrix} 
 \mathbb{I} & A^1 \\
 \mathbb{O} & \mathbb{I} \\
\end{pmatrix}
\begin{pmatrix}
 q \\
 p \\
\end{pmatrix}
+ b ,
```
 
or 
 
```math
\mathcal{L}^{w,\mathrm{low}}
\begin{pmatrix}  q  \\  
 p  \end{pmatrix} =  
  \begin{pmatrix} 
 \mathbb{I} & \mathbb{O}/A^w  \\ 
 A^w/\mathbb{O} & \mathbb{I}
 \end{pmatrix} \cdots 
  \begin{pmatrix} 
 \mathbb{I} & A^2  \\ 
 \mathbb{O} & \mathbb{I}
 \end{pmatrix}
 \begin{pmatrix} 
 \mathbb{I} & \mathbb{O}  \\ 
 A^1 & \mathbb{I}
 \end{pmatrix}
 \begin{pmatrix}  q  \\  
 p  \end{pmatrix}
  + b . 
```

The superscripts ``\mathrm{up}`` and ``\mathrm{low}`` indicate whether the ``q`` or the ``p`` part is first changed[^3]. The learnable parameters are the symmetric matrices ``A^i\in\mathcal{S}_\mathrm{sym}(d)`` and the bias ``b\in\mathbb{R}^{2d}``. The integer ``w`` is the number of linear layers in one block. It can be shown that five of these layers, i.e. ``w\geq{}5``, can represent any linear symplectic map [jin2022optimal](@cite), so ``w`` need not be larger than five. We denote the set of symplectic linear layers by ``\mathcal{M}^L``.

[^3]: "up" means we first change the ``q`` part and "low" means we first change the ``p`` part. This can be set via the keyword `init_upper_linear` in [`LASympNet`](@ref). 

The second type of layer needed for ``LA``-SympNets are [activation layers](@ref "SympNet Gradient Layer").
 
An ``LA``-SympNet is a mapping of the form ``\Psi=l_{k} \circ a_{k} \circ l_{k-1} \circ \cdots \circ a_1 \circ l_0`` where ``(l_i)_{0\leq i\leq k} \subset \mathcal{M}^L`` and ``(a_i)_{1\leq i\leq k} \subset \mathcal{M}^A``. We will refer to ``k`` as the *number of hidden layers* of the SympNet[^4] and the number ``w`` above as the *depth* of the linear layer.

[^4]: Note that if ``k=0`` then the ``LA``-SympNet consists of only one linear layer.

We give an example of calling ``LA``-SympNet:

```@example
using GeometricMachineLearning

k = 1
w = 2
arch = LASympNet(4; 
                    nhidden = k, 
                    depth = 2, 
                    init_upper_linear = true, 
                    init_upper_act = true, 
                    activation = tanh)

model = Chain(arch).layers
```
 
The keywords `init_upper_linear` and `init_upper_act` indicate whether the first linear (respectively activation) layer is of ``q`` type[^5].

[^5]: Similarly to `init_upper_linear`, if `init_upper_act = true` then the first activation layer is of ``q`` type, i.e. changes the ``q`` component and leaves the ``p`` component unchanged. 

## ``G``-SympNets
 
``G``-SympNets are an alternative to ``LA``-SympNets. They are built with only one kind of layer, the [gradient layer](@ref "SympNet Gradient Layer"). If we denote by ``\mathcal{M}^G`` the set of gradient layers, a ``G``-SympNet is a function of the form ``\Psi=g_k \circ g_{k-1} \circ \cdots \circ g_1`` where ``(g_i)_{1\leq i\leq k} \subset \mathcal{M}^G``. The index ``k`` here is the *number of layers* in the SympNet.

```@example
using GeometricMachineLearning

k = 2
n = 10
arch = GSympNet(4; upscaling_dimension = n, n_layers = k, init_upper = true, activation = tanh)

model = Chain(arch).layers
```

The keyword `init_upper` for [`GSympNet`](@ref) is similar as in the case for [`LASympNet`](@ref). The keyword `upscaling_dimension` is explained in the section on the [SympNet gradient layer](@ref "SympNet Gradient Layer").

## Universal Approximation Theorems

In order to state the *universal approximation theorem* for both architectures we first need a few definitions:
 
Let ``U`` be an open set of ``\mathbb{R}^{2d}``, and let us denote by ``\mathcal{SP}^r(U)`` the set of ``C^r`` smooth symplectic maps on ``U``. We now define a topology on ``C^r(K, \mathbb{R}^{2d})``, the set of ``C^r``-smooth maps from a compact set ``K\subset{}U`` to ``\mathbb{R}^{2d}`` through the norm

```math
||f||_{C^r(K,\mathbb{R}^{2d})} = \sum_{|\alpha|\leq r} \underset{1\leq i \leq 2d}{\max} \hspace{2mm} \underset{x\in K}{\sup} |D^\alpha f_i(x)|,
```
where the differential operator ``D^\alpha`` is defined by 
```math
D^\alpha f = \frac{\partial^{|\alpha|} f}{\partial x_1^{\alpha_1}...x_n^{\alpha_n}},
```
with ``|\alpha| = \alpha_1 +...+ \alpha_{2d}``. We impose the following condition (``r`` finiteness) on the activation function:

```@eval
Main.definition(raw"``\sigma`` is **``r``-finite** if ``\sigma\in C^r(\mathbb{R},\mathbb{R})`` and ``\int |D^r\sigma(x)|dx <\infty``.")
```

We further consider the topology on ``C^r(U, \mathbb{R}^d)`` induced by ``||\cdot ||_{C^r(\cdot, \mathbb{R}^d)}`` and the associated notion of [denseness](@ref "Basic Concepts from General Topology"):

```@eval
Main.definition(raw"Let ``m,d,r\in \mathbb{N}`` with ``m,d>0`` be given, ``U`` an open subset of ``\mathbb{R}^m``, and ``I,J\subset C^r(U,\mathbb{R}^d)``. We say ``J`` is **``r``-uniformly dense on compacta in ``I``** if ``J \subset I`` and for any ``f\in I``, ``\epsilon>0``, and any compact ``K\subset U``, there exists ``g\in J`` such that ``||f-g||_{C^r(K,\mathbb{R}^{d})} < \epsilon``.")
```

```@eval
Main.remark(raw"The associated topology to this notion of denseness is the **compact-open topology**. It is generated by the following sets:
" * Main.indentation * raw"```math
" * Main.indentation * raw" V(K, U) := \{f\in{}C^r(\mathbb{R}^m, \mathbb{R}^d): \text{ such that $f(K)\subset{}U$}\},
" * Main.indentation * raw"```
" * Main.indentation * raw"i.e. the *compact-open topology* is the smallest topology that contains all sets of the form ``V(K, U)``.")
```

We can now state the universal approximation theorems:

```@eval
Main.theorem(raw"For any positive integer ``r>0`` and open set ``U\in \mathbb{R}^{2d}``, the set of ``LA``-SympNet is ``r``-uniformly dense on compacta in ``SP^r(U)`` if the activation function ``\sigma`` is ``r``-finite."; name = raw"Approximation theorem for LA-SympNets")
```

and

```@eval
Main.theorem(raw"For any positive integer ``r>0`` and open set ``U\in \mathbb{R}^{2d}``, the set of ``G``-SympNet is ``r``-uniformly dense on compacta in ``SP^r(U)`` if the activation function ``\sigma`` is ``r``-finite."; name = raw"Approximation theorem for G-SympNets")
```

There are many ``r``-finite activation functions commonly used in neural networks, for example:
- The sigmoid activation function: ``\sigma(x) = {1} / (1+e^{-x})``, 
- The hyperbolic tangent function: ``\tanh(x) = (e^x-e^{-x}) / (e^x+e^{-x})``. 

The universal approximation theorems state that we can, in principle, get arbitrarily close to any symplectomorphism defined on ``\mathbb{R}^{2d}``. But this does not tell us anything about how to optimize the network. This is can be done with any common [neural network optimizer](@ref "Neural Network Optimizers") and these neural network optimizers always rely on a corresponding loss function.  

## Loss function

To train the SympNet, one needs data along a trajectory such that the model is trained to perform an integration. The loss function is defined as[^6]:

[^6]: This loss function is implemented as [`FeedForwardLoss`](@ref) in `GeometricMachineLearning`.

```math
\mathrm{loss}(z^\mathrm{c}, z^\mathrm{p}) = \frac{|| z^\mathrm{c} - z^\mathrm{p} ||}{|| z^\mathrm{c} ||},
```

where 

```math
z^\mathrm{c} = \begin{pmatrix} q^\mathrm{c} \\ p^\mathrm{c} \end{pmatrix}
``` 

is the current state and 

```math
z^\mathrm{p} = \begin{pmatrix} q^\mathrm{p} \\ p^\mathrm{p} \end{pmatrix}
```

is the predicted state. In the [example section](@ref "SympNets with `GeometricMachineLearning`") we show how to use SympNets in `GeometricMachineLearning.jl` and how to [modify the loss function](@ref "Adjusting the Loss Function").

## Library Functions

```@docs
SympNet
LASympNet
GSympNet
```

```@raw latex
\begin{comment}
```

## References

```@bibliography
Pages = []
Canonical = false

jin2020sympnets
```

```@raw latex
\end{comment}
```