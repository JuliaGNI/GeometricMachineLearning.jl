# Symplectic Attention

There are two ways of constructing symplectic attention: with a matrix-like softmax and a classical softmax.

Both of these approaches are however based on taking the derivative of an expression with respect to its input arguments (similar to [gradient layers](@ref "SympNet Gradient Layer")).

We first start with singlehead attention[^1]. We start with the [correlation matrix](@ref "The Attention Layer"):

[^1]: The multihead attention case is in principle not much more difficult, but care has to be taken with regards to the projection spaces.

```math
C = Z^TAZ \implies C_{mn} = \sum_{k\ell}Z_{km}A_{k\ell}Z_{\ell{}n}.
```

Its element-wise derivative is:

```math
\frac{\partial{}C_{mn}}{\partial{}Z_{ij}} = \sum_{\ell}(\delta_{jm}A_{i\ell}Z_{\ell{}n} + \delta_{jn}X_{\ell{}m}A_{\ell{}i}).
```

## Matrix Softmax

Here we take as a staring point the expression:

```math
\Sigma(Z) = \mathrm{log}(1 + \exp(\sum_{m,n}C_{mn})).
```

Its gradient (with respect to ``Z``) is:

```math
\frac{\partial\Sigma(Z)}{\partial{}Z_{ij}} = \frac{1}{1 + \sum{m, n}\exp(C_{mn})}\sum_{m'n'}\exp(C_{m'n'})\sum_{\ell}(\delta_{jm'}A_{i\ell}Z_{\ell{}n'} + \delta_{jn'}X_{\ell{}m'}A_{\ell{}i}) = \frac{1}{1 + \sum_{m,n}\exp(C_{mn})}\{[AX\exp.(C)^T]_{ij} +  [A^TX\exp.(C)]_{ij}\}.
```

Note that if `A` is a [`SymmetricMatrix`](@ref) the expression than simplifies to:

```math
\frac{\partial\Sigma(Z)}{\partial{}Z_{ij}} = 2\frac{1}{1 + \sum_{m,n}\exp(C_{mn})}[AX\exp.(C)^T]_{ij},
```

or written in matrix notation:

```math
\nabla_Z\Sigma(Z) = 2\frac{1}{1 + \sum_{m,n}\exp(C_{mn})}AX\exp.(C).
```

Whether we use a [`SymmetricMatrix`](@ref) for ``A`` or not can be set with the keyword `symmetric` in [`SymplecticAttention`](@ref).

## Vector Softmax

Here we take as a staring point the expression:

```math
\Sigma(Z) = \sum_{n}\mathrm{log}(1 + \exp(\sum_{m}C_{mn})).
```

We then get:

```math
\frac{\partial\Sigma(Z)}{\partial{}Z_{ij}} = \sum_n\frac{\exp(C_{jn})}{1 + \sum_m\exp(C_{mn})}[AZ]_{in} + \frac{1}{1 + \sum_m\exp(C_{mj})}[A^TZ\exp(C)]_{ij}.
```

The second term in this expression is equivalent to a *standard attention step*:

```math
\mathrm{TermII:}\qquad A^TZ\mathrm{softmax}(C).
```

The first term is equivalent to:

```math
\mathrm{TermI:}\qquad \sum_n [AZ]_{in}[\mathrm{softmax}(C)^T]_{nj} \equiv AZ(\mathrm{softmax}(C))^T.
```

If we again assume that the matrix `A` is a [`SymmetricMatrix`](@ref) then the expression simplifies to:

```math
\nabla_Z\Sigma(Z) = AZ\mathrm{softmax}(C).
```

## Library Functions

```@docs
SymplecticAttention
SymplecticAttentionQ
SymplecticAttentionP
```