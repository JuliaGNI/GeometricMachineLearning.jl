# Pullbacks and Automatic Differentiation

Automatic Differentiation is an important part of modern machine learning libraries. It is essentially a tool to compute the gradient of a loss function with respect to its input arguments. 

## How to Compute Pullbacks

`GeometricMachineLearning` has many pullbacks for custom array types and other operations implemented. The need for this essentially comes from the fact that we cannot trivially differentiate custom GPU kernels at the moment[^1].

[^1]: This will change once we switch to Enzyme (see [moses2021reverse](@cite)), but the package is still in its infancy. 

## What is a pullback?

Here we first explain the principle of a pullback with the example of a vector-valued function. The generalization to matrices and higher-order tensors is straight-forward. 

The pullback of a vector-valued function ``f:\mathbb{R}^{n}\to\mathbb{R}^m`` can be interpreted as the *sensitivities in the input space* ``\mathbb{R}^n`` with respect to variations in the output space ``\mathbb{R}^m`` via the function ``f``: 

```math 
\left[\mathrm{pullback}(f)[a\in\mathbb{R}^n, db\in\mathbb{R}^m]\right]_i = \sum_{j=1}^m\frac{\partial{}f_j}{\partial{}a_i}db_j.
```

This principle can easily be generalized to matrices. For this consider the function ``g::\mathbb{R}^{n_1\times{}n_2}\to\mathbb{R}^{m_1\times{}m_2}``. For this case we have: 

```math
\left[\mathrm{pullback}(g)[A\in\mathbb{R}^{n_1\times{}n_2}, dB\in\mathbb{R}^{m_1\times{}m_2}]\right]_{(i_1, i_2)} = \sum_{j_1=1}^{m_1}\sum_{j_2=1}^{m_2}\frac{\partial{}f_{(j_1, j_2)}}{\partial{}a_{(i_1, i_2)}}db_{(j_1, j_2)}.
```

The generalization to higher-order tensors is again straight-forward.

### Illustrative example 

Consider the matrix inverse ``\mathrm{inv}: \mathbb{R}^{n\times{}n}\to\mathbb{R}^{n\times{}n}`` as an example. This fits into the above framework where ``inv`` is a matrix-valued function from ``\mathbb{R}^{n\times{}n}`` to ``\mathbb{R}^{n\times{}n}``. We here write ``B := A^{-1} = \mathrm{inv}(A)``. We thus have to compute: 

```math 
\left[\mathrm{pullback}(\mathrm{inv})[A\in\mathbb{R}^{n\times{}n}, dB\in\mathbb{R}^{n\times{}n}]\right]_{(i, j)} = \sum_{k=1}^{n}\sum_{\ell=1}^{n}\frac{\partial{}b_{k, \ell}}{\partial{}a_{i, j}}db_{k, \ell}.
```

For a matrix ``A`` that depends on a parameter ``\varepsilon`` we have that: 
```math
\frac{\partial}{\partial\varepsilon}B = -B\left( \frac{\partial}{\partial\varepsilon} \right) B.
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

## Motivation from a differential-geometric perspective 

The notions of a pullback in automatic differentiation and differential geometry are closely related (see e.g. [betancourt2018geometric](@cite) and [bolte2020mathematical](@cite)). In both cases we want to compute, based on a mapping ``f:\mathcal{V}\to\mathcal{W}, a \mapsto f(a) =: b``, a *map of differentials* ``db \mapsto da``. In the differential geometry case ``db`` and ``da`` are part of the associated cotangent spaces, i.e. ``db\in{}T^*_b\mathcal{W}`` and ``da\in{}T^*_a\mathcal{V}``; in AD we (mostly) deal with spaces of arrays, i.e. vector spaces, which means that ``db\in\mathcal{W}`` and ``da\in\mathcal{V}``.

```@bibliography 
Pages = []
Canonical = false

betancourt2018geometric
bolte2020mathematical 
```