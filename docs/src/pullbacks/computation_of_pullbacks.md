# Pullbacks and Automatic Differentiation

Automatic Differentiation is an important part of modern machine learning. It is essentially a tool to compute the gradient of a loss function with respect to its input arguments, i.e. given a function ``L:\Theta\to\mathbb{R}`` an AD routine computes:

```math
    \mathrm{AD}: \theta \mapsto \nabla_\theta{}L.
```

When we train a neural network the function ``L`` is the composition of a neural network 

```math
    \mathcal{NN}_\theta(x) = l^{(n)}_{\theta_n}\circ{}l^{(1)}_{\theta_1}(x)
```

and a [loss function](@ref "Different Neural Network Losses"), so we have:

```math
    L(\theta) = \mathtt{loss}(l^{(n)}_{\theta_n}\circ{}l^{(1)}_{\theta_1}(x), \mathtt{minibatch})
```

where ``\theta = (\theta_1, \ldots, \theta_n)`` and ``L`` further depends on a specific mini batch here. So what we need to do is compute the pullback[^1] of every single layer ``l^{(i)}_{\theta_i}``. For this consider a function[^2] ``f: \mathbb{R}^m \to \mathbb{R}^n`` and ``x\in\mathbb{R}^m``. The pullback of ``f`` is a function that, depending on ``x``, provides a recipe to map an element from ``\mathbb{R}^n`` to an element of ``\mathbb{R}^m``:

[^1]: The term *pullback* originally comes from differential geometry [bishop1980tensor](@cite). This motivation is discussed below.
[^2]: ``\mathbb{R}^m`` can be interpreted as the space of the neural network parameters ``\Theta`` here. 

```math
    \mathrm{pullbak}(f)[x]:\mathbb{R}^m \simeq T^*_{f(x)}\mathbb{R}^m \to T^*_{x}\mathbb{R}^n \simeq \mathbb{R}^n,
```

where ``T^*_x\mathcal{V}`` is the cotangent space of ``\mathcal{V}`` at ``x``. 

## How to Compute Pullbacks

`GeometricMachineLearning` has many pullbacks for custom array types and other operations implemented. The need for this essentially comes from the fact that we cannot trivially differentiate custom GPU kernels[^3]. Implemented custom pullback comprise [parallel multiplications with tensors](@ref "Tensors in `GeometricMachineLearning`").

[^3]: This may change in the future if the package `Enzyme` [moses2021reverse](@cite) reaches maturity.

## What is a Pullback?

Here we first explain the principle of a pullback with the example of a vector-valued function. The generalization to matrices and higher-order tensors is straight-forward. 

The pullback of a vector-valued function ``f:\mathbb{R}^{n}\to\mathbb{R}^m`` can be interpreted as the *sensitivities in the input space* ``\mathbb{R}^n`` with respect to variations in the output space ``\mathbb{R}^m`` via the function ``f``: 

```math 
\left[\mathrm{pullback}(f)[a\in\mathbb{R}^n, db\in\mathbb{R}^m]\right]_i = \sum_{j=1}^m\frac{\partial{}f_j}{\partial{}a_i}db_j.
```

This principle can easily be generalized to matrices. For this consider the function ``g::\mathbb{R}^{n_1\times{}n_2}\to\mathbb{R}^{m_1\times{}m_2}``. We then have: 

```math
\left[\mathrm{pullback}(g)[A\in\mathbb{R}^{n_1\times{}n_2}, dB\in\mathbb{R}^{m_1\times{}m_2}]\right]_{(i_1, i_2)} = \sum_{j_1=1}^{m_1}\sum_{j_2=1}^{m_2}\frac{\partial{}f_{(j_1, j_2)}}{\partial{}a_{(i_1, i_2)}}db_{(j_1, j_2)}.
```

The generalization to higher-order tensors is again straight-forward.

### Illustrative example 

Consider the matrix inverse ``\mathrm{inv}: \mathbb{R}^{n\times{}n}\to\mathbb{R}^{n\times{}n}`` as an example. This fits into the above framework where ``inv`` is a matrix-valued function from ``\mathbb{R}^{n\times{}n}`` to ``\mathbb{R}^{n\times{}n}``. We here write ``B := A^{-1} = \mathrm{inv}(A)``. We thus have to compute: 

```math 
\left[\mathrm{pullback}(\mathrm{inv})[A\in\mathbb{R}^{n\times{}n}, dB\in\mathbb{R}^{n\times{}n}]\right]_{(i, j)} = \sum_{k=1}^{n}\sum_{\ell=1}^{n}\frac{\partial{}b_{k, \ell}}{\partial{}a_{i, j}}db_{k, \ell}.
```

For a matrix ``A`` that depends on a parameter ``\varepsilon`` we have: 
```math
\frac{\partial}{\partial\varepsilon}B = -B\left( \frac{\partial}{\partial\varepsilon} A \right) B.
```

This can easily be checked: 
```math 
\mathbb{O} = \frac{\partial}{\partial\varepsilon}\mathbb{I} = \frac{\partial}{\partial\varepsilon}(AB) = A\frac{\partial}{\partial\varepsilon}B + \left(\frac{\partial}{\partial\varepsilon}A\right)B.
```

We can then write: 

```math
\begin{aligned}
\sum_{k,\ell}\left( \frac{\partial}{\partial{}a_{ij}} b_{k\ell} \right) db_{k\ell}  & = \sum_{k\ell}\left[ \frac{\partial}{\partial{}a_{ij}} B \right]_{k\ell} db_{k,\ell} \\ 
& = - \sum_{k,\ell}\left[B \left(\frac{\partial}{\partial{}a_{ij}} A\right) B \right]_{k\ell} db_{k\ell} \\ 
& = - \sum_{k,\ell,m,n}b_{km} \left(\frac{\partial{}a_{mn}}{\partial{}a_{ij}}\right) b_{n\ell} db_{k\ell} \\ 
& = - \sum_{k,\ell,m,n}b_{km} \delta_{im}\delta_{jn} b_{n\ell} db_{k\ell} \\ 
& = - \sum_{k,\ell}b_{ki} b_{j\ell} db_{k\ell} \\ 
& \equiv - B^T\cdot{}dB\cdot{}B^T. 
\end{aligned}
```

We use this expression to differentiate the [`VolumePreservingAttention`](@ref) layer. 

## Motivation from a differential-geometric perspective 

The notion of a *pullback in automatic differentiation* is motivated by the concept of *pullback in differential geometry* [betancourt2018geometric, bolte2020mathematical](@cite). In both cases we want to compute, based on a mapping 

```math
f:\mathcal{V}\to\mathcal{W}, a \mapsto f(a) =: b, 
```
a *map of differentials* ``db \mapsto da``. In the differential geometry case ``db`` and ``da`` are part of the associated cotangent spaces, i.e. ``db\in{}T^*_b\mathcal{W}`` and ``da\in{}T^*_a\mathcal{V}``; in AD we (mostly) deal with spaces of arrays, i.e. vector spaces, which means that ``T^*_b\mathcal{W} \simeq \mathcal{W}`` and ``T^*_a\mathcal{V} \simeq \mathcal{V}``. If we have neural network weights on manifolds however, then we have to map weights from ``T^*_a\mathcal{V}`` (the result of an AD routine) to ``T_a\mathcal{V}`` before we can apply a [retraction](@ref "Retractions"). The mapping 

```math
T^*_a\mathcal{V} \to T_a\mathcal{V}
```

is equivalent to applying the [Riemannian gradient](@ref "The Riemannian Gradient").

## Library Functions 

```@docs
GeometricMachineLearning.ZygotePullback
```

```@raw latex
\begin{comment}
```

## References


```@raw latex
\end{comment}
```


```@raw html
<!--
```

# References


```@raw html
-->
```

```@bibliography 
Pages = []
Canonical = false

betancourt2018geometric
bolte2020mathematical 
```