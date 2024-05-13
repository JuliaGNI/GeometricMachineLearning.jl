# SympNet Gradient Layer

The Sympnet gradient layer (called [`GradientLayer`](@ref) in `GeometricMachineLearning`) is based on the following theorem: 

```@eval
Main.theorem(raw"""Given a symplectic vector space ``\mathbb{R}^{2n}`` which coordinates ``q_1, \ldots, q_n, p_1, \ldots, p_n`` and a function ``f:\mathbb{R}^n\to\mathbb{R}`` that only acts on the ``q`` part, the map ``(q, p) \mapsto (q, p + \nabla_qf)`` is symplectic. A similar statement holds if ``f`` only acts on the ``p`` part.""")
```

Proofing this is straightforward by looking at the gradient of the mapping:

```math
    \begin{pmatrix}
        \mathbb{I} & \mathbb{O} \\ 
        \nabla_q^2f & \mathbb{I}
    \end{pmatrix},
```

where ``\nabla_q^2f`` is the Hessian of ``f``. This matrix is symmetric and for any symmetric matrix ``A`` we have that: 

```math
    \begin{pmatrix}
        \mathbb{I} & \mathbb{O} \\ 
        A & \mathbb{I}
    \end{pmatrix}^T \mathbb{J}_{2n} 
    \begin{pmatrix} 
        \mathbb{I} & \mathbb{O} \\ 
        A & \mathbb{I} 
    \end{pmatrix} = 
    \begin{pmatrix}
        \mathbb{I} & A \\ 
        \mathbb{O} & \mathbb{I}
    \end{pmatrix} 
    \begin{pmatrix} 
        \mathbb{O} & \mathbb{I} \\ 
        -\mathbb{I} & \mathbb{O} 
    \end{pmatrix} 
    \begin{pmatrix}
        \mathbb{I} & \mathbb{O} \\ 
        A & \mathbb{I}
    \end{pmatrix} = 
    \begin{pmatrix}
        \mathbb{O} & \mathbb{I} \\ 
        -\mathbb{I} & \mathbb{O} 
    \end{pmatrix}.
```

If we deal with [`GSympNet`](@ref)s this function ``f`` is 

```math
    f(q) = a^T \Sigma(Kq + b),
```

where ``a, b\in\mathbb{R}^m``, ``K\in\mathbb{R}^{m\times{}n}`` and ``\Sigma`` is the antiderivative of some common activation function ``\sigma``. We routinely refer to ``m`` as the *upscaling dimension* in `GeometricMachineLearning`. Computing the gradient of ``f`` gives: 

```math
    [\nabla_qf]_k = \sum_{i=1}^m a_i \sigma(\sum_{j=1}^nk_{ij}q_j + b_i)k_{ik} = = K^T a \odot \sigma(Kq + b),
```

where ``\odot`` is the element-wise product, i.e. ``[a\odot{}v]_k = a_kv_k``.