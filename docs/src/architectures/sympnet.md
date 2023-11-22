# SympNet

This document discusses the SympNet architecture and its implementation in `GeometricMachineLearning.jl`.

## Quick overview of the theory of SympNet

### Principle

SympNets (see [jin2020sympnets](@cite) for the eponymous paper) are a type of neural network that can model the trajectory of a Hamiltonian system in phase space. Take $(q^T,p^T)^T=(q_1,\ldots,q_d,p_1,\ldots,p_d)^T\in \mathbb{R}^{2d}$ as the coordinates in phase space, where $q=(q_1, \ldots, q_d)^T\in \mathbb{R}^{d}$ is refered to as the *position* and $p=(p_1, \ldots, p_d)^T\in \mathbb{R}^{d}$ the *momentum*. Given a point $(q^T,p^T)^T$ in $\mathbb{R}^{2d}$ the SympNet aims to compute the *next position* $((q')^T,(p')^T)^T$ and thus predicts the trajectory while preserving the *symplectic structure* of the system.
SympNets are enforcing symplecticity strongly, meaning that this property is hard-coded into the network architecture. The layers are reminiscent of traditional neural network feedforward layers, but have a strong restriction imposed on them in order to be symplectic.

SympNets can be viewed as a "symplectic integrator" (see [hairer2006geometric](@cite) and [leimkuhler2004simulating](@cite)). Their goal is to predict, based on an initial condition $((q^{(0)})^T,(p^{(0)})^T)^T$, a sequence of points in phase space that fit the training data as well as possible:
```math
\begin{pmatrix} q^{(0)} \\ p^{(0)} \end{pmatrix}, \cdots, \begin{pmatrix} \tilde{q}^{(1)} \\ \tilde{p}^{(1)} \end{pmatrix}, \cdots \begin{pmatrix} \tilde{q}^{(n)} \\ \tilde{p}^{(n)} \end{pmatrix}.
```
The tilde in the above equation indicates *predicted data*. The time step between predictions is not a parameter we can choose but is related to the *temporal frequency of the training data*. This means that if data is recorded in an interval of e.g. 0.1 seconds, then this will be the time step of our integrator.

### Architecture of SympNets
![](../tikz/sympnet_architecture.png)

There are two types of SympNet architectures: $LA$-SympNets and $G$-SympNets. 
 
#### $LA$-SympNet

The first type of SympNets, $LA$-SympNets, are obtained from composing two types of layers: *symplectic linear layers* and *symplectic activation layers*.  For a given integer $n$, a symplectic linear layer is defined by

```math
\mathcal{L}^{n,q}
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
\mathcal{L}^{n,p}
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

The superscripts $q$ and $p$ indicate whether the $q$ or the $p$ part is changed. The learnable parameters are the symmetric matrices $S^i\in\mathbb{R}^{d\times d}$ and the bias $b\in\mathbb{R}^{2d}$. The integer $n$ is the width of the symplectic linear layer. It can be shown that five of these layers, i.e. $n\geq{}5$, can represent any linear symplectic map (see [jin2022optimal](@cite)), so $n$ need not be larger than five. We denote the set of symplectic linear layers by $\mathcal{M}^L$.

The second type of layer needed for $LA$-SympNets are so-called *activation layers*:

```math
 \mathcal{A}^{q}  \begin{pmatrix}  q  \\  
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
 
 and
 
```math
 \mathcal{A}^{p}  \begin{pmatrix}  q  \\  
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
The activation function $\sigma$ can be any nonlinearity (on which minor restrictions are imposed below). Here the *scaling vector* $a\in\mathbb{R^{d}}$ constitutes the learnable weights. We denote the set of symplectic activation layers by $\mathcal{M}^A$. 
 
An $LA$-SympNet is a function of the form $\Psi=l_{k} \circ a_{k} \circ l_{k-1} \circ \cdots \circ a_1 \circ l_0$ where $(l_i)_{0\leq i\leq k} \subset (\mathcal{M}^L)^{k+1}$ and $(a_i)_{1\leq i\leq k} \subset (\mathcal{M}^A)^{k}$. We will refer to $k$ as the *number of hidden layers* of the SympNet[^1] and the number $n$ above as the *depth* of the linear layer.

[^1]: Note that if $k=1$ then the $LA$-SympNet consists of only one linear layer.
 
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

The parameters of this layer are the *scaling matrix* $K\in\mathbb{R}^{m\times d}$, the bias $b\in\mathbb{R}^{m}$ and the *scaling vector* $a\in\mathbb{R}^{m}$. The name "gradient layer" has its origin in the fact that the expression $[K^T\mathrm{diag}(a)\sigma(Kq+b)]_i = \sum_jk_{ji}a_j\sigma(\sum_\ell{}k_{j\ell}q_\ell+b_j)$ is the gradient of a function $\sum_ja_j\tilde{\sigma}(\sum_\ell{}k_{j\ell}q_\ell+b_j)$, where $\tilde{\sigma}$ is the antiderivative of $\sigma$. The first dimension of $K$ we refer to as the *upscaling dimension*.
 
If we denote by $\mathcal{M}^G$ the set of gradient layers, a $G$-SympNet is a function of the form $\Psi=g_k \circ g_{k-1} \circ \cdots \circ g_0$ where $(g_i)_{0\leq i\leq k} \subset (\mathcal{M}^G)^k$. The index $k$ is again the *number of hidden layers*.

Further note here the different roles played by round and square brackets: the latter indicates a nonlinear operation as opposed to a regular vector or matrix. 

### Universal approximation theorems

In order to state the *universal approximation theorem* for both architectures we first need a few definitions:
 
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

The universal approximation theorems state that we can, in principle, get arbitrarily close to any symplectomorphism defined on $\mathbb{R}^{2d}$. But this does not tell us anything about how to optimize the network. This is can be done with any common [neural network optimizer](../Optimizer.md) and these neural network optimizers always rely on a corresponding loss function.  

## Loss function

To train the SympNet, one need data along a trajectory such that the model is trained to perform an integration. These data are $(Q,P)$ where $Q[i,j]$ (respectively $P[i,j]$) is the real number $q_j(t_i)$ (respectively $p[i,j]$) which is the j-th coordinates of the generalized position (respectively momentum) at the i-th time step. One also need a loss function defined as :

$$Loss(Q,P) = \underset{i}{\sum} d(\Phi(Q[i,-],P[i,-]), [Q[i,-] P[i,-]]^T)$$
where $d$ is a distance on $\mathbb{R}^d$.

See the [tutorial section](../tutorials/sympnet_tutorial.md) for an introduction into using SympNets with `GeometricMachineLearning.jl`.

## References
```@bibliography
Pages = []
Canonical = false

jin2020sympnets
```