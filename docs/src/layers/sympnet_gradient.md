# SympNet Layers

The *SympNet paper* [jin2020sympnets](@cite) discusses three different kinds of sympnet layers: *activation layers*, *linear layers* and *gradient layers*. We discuss them below. Because activation layers are just a simplified form of gradient layers those are introduced together. A neural network that consists of many of these layers we call a [SympNet](@ref "SympNet Architecture").

## SympNet Gradient Layer

The Sympnet gradient layer is based on the following theorem: 

```@eval
Main.theorem(raw"""Given a symplectic vector space ``\mathbb{R}^{2n}`` with coordinates ``q_1, \ldots, q_n, p_1, \ldots, p_n`` and a function ``f:\mathbb{R}^n\to\mathbb{R}`` that only acts on the ``q`` part, the map ``(q, p) \mapsto (q, p + \nabla_qf)`` is symplectic. A similar statement holds if ``f`` only acts on the ``p`` part.""")
```

```@eval
Main.proof(raw"""Proofing this is straightforward by looking at the gradient of the mapping:
""" * Main.indentation * raw"""
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw"""    \begin{pmatrix}
""" * Main.indentation * raw"""        \mathbb{I} & \mathbb{O} \\ 
""" * Main.indentation * raw"""        \nabla_q^2f & \mathbb{I}
""" * Main.indentation * raw"""    \end{pmatrix},
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""where ``\nabla_q^2f`` is the Hessian of ``f``. This matrix is symmetric and for any symmetric matrix ``A`` we have that: 
""" * Main.indentation * raw"""```math
""" * Main.indentation * raw""" \begin{pmatrix}
""" * Main.indentation * raw"""     \mathbb{I} & \mathbb{O} \\ 
""" * Main.indentation * raw"""     A & \mathbb{I}
""" * Main.indentation * raw""" \end{pmatrix}^T \mathbb{J}_{2n} 
""" * Main.indentation * raw""" \begin{pmatrix} 
""" * Main.indentation * raw"""     \mathbb{I} & \mathbb{O} \\ 
""" * Main.indentation * raw"""     A & \mathbb{I} 
""" * Main.indentation * raw""" \end{pmatrix} = 
""" * Main.indentation * raw""" \begin{pmatrix}
""" * Main.indentation * raw"""     \mathbb{I} & A \\ 
""" * Main.indentation * raw"""     \mathbb{O} & \mathbb{I}
""" * Main.indentation * raw""" \end{pmatrix} 
""" * Main.indentation * raw""" \begin{pmatrix} 
""" * Main.indentation * raw"""     \mathbb{O} & \mathbb{I} \\ 
""" * Main.indentation * raw"""     -\mathbb{I} & \mathbb{O} 
""" * Main.indentation * raw""" \end{pmatrix} 
""" * Main.indentation * raw""" \begin{pmatrix}
""" * Main.indentation * raw"""     \mathbb{I} & \mathbb{O} \\ 
""" * Main.indentation * raw"""     A & \mathbb{I}
""" * Main.indentation * raw""" \end{pmatrix} = 
""" * Main.indentation * raw""" \begin{pmatrix}
""" * Main.indentation * raw"""     \mathbb{O} & \mathbb{I} \\ 
""" * Main.indentation * raw"""     -\mathbb{I} & \mathbb{O} 
""" * Main.indentation * raw""" \end{pmatrix} = \mathbb{J}_{2n},
""" * Main.indentation * raw"""```
""" * Main.indentation * raw"""thus showing symplecticity.""")
```

If we deal with [`GSympNet`](@ref)s this function ``f`` is 

```math
    f(q) = a^T \Sigma(Kq + b),
```

where ``a, b\in\mathbb{R}^m``, ``K\in\mathbb{R}^{m\times{}n}`` and ``\Sigma`` is the antiderivative of some common activation function ``\sigma``. We routinely refer to ``m`` as the *upscaling dimension* in `GeometricMachineLearning`. Computing the gradient of ``f`` gives: 

```math
    [\nabla_qf]_k = \sum_{i=1}^m a_i \sigma(\sum_{j=1}^nk_{ij}q_j + b_i)k_{ik} = K^T (a \odot \sigma(Kq + b)),
```

where ``\odot`` is the element-wise product, i.e. ``[a\odot{}v]_k = a_kv_k``. This is the form that *gradient layers* take. In addition to gradient layers `GeometricMachineLearning` also has *linear* and *activation* layers implemented. Activation layers are simplified versions of *gradient layers*. These are equivalent to taking ``m = n`` and ``K = \mathbb{I}.``

## SympNet Linear Layer

Linear layers of type ``p`` are of the form:

```math
\begin{pmatrix} q \\ p \end{pmatrix} \mapsto \begin{pmatrix} \mathbb{I} & \mathbb{O} \\ A & \mathbb{I} \end{pmatrix} \begin{pmatrix} q \\ p \end{pmatrix},
```

where ``A`` is a symmetric matrix. This is implemented very efficiently in `GeometricMachineLearning` with the special matrix [`SymmetricMatrix`](@ref).

## Library Functions

```@docs
GeometricMachineLearning.SympNetLayer
GeometricMachineLearning.GradientLayer
GradientLayerQ
GradientLayerP
GeometricMachineLearning.LinearLayer
LinearLayerQ
LinearLayerP
GeometricMachineLearning.ActivationLayer
ActivationLayerQ
ActivationLayerP
```

## References

```@bibliography
Canonical = false
Pages = []

jin2020sympnets
```