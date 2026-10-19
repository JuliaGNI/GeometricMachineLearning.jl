@doc raw"""
    VolumePreservingFeedForwardLayer <: AbstractExplicitLayer

The volume-preserving feedforward layer. The layers do the following:

```math
x \mapsto \begin{cases} \sigma(Lx + b) & \text{where $L$ is }\mathtt{StrictlyLowerTriangular}, \\ \sigma(Ux + b) & \text{where $U$ is }\mathtt{StrictlyUpperTriangular}. \end{cases}
```

The functor can be applied to a vector, a matrix or a tensor. The special matrices are implemented as [`StrictlyLowerTriangular`](@extref GeometricOptimizers GeometricOptimizers.StrictlyLowerTriangular) and [`StrictlyUpperTriangular`](@extref GeometricOptimizers GeometricOptimizers.StrictlyUpperTriangular).

# Implementation

One type, with the triangle as its last type parameter: `:lower` for the layer the constructor
[`VolumePreservingLowerLayer`](@ref) makes and `:upper` for the one [`VolumePreservingUpperLayer`](@ref)
makes. The two names are aliases of this type, so a signature or a `@docs` block can name either.
"""
struct VolumePreservingFeedForwardLayer{M, N, UseBias, AT, Triangle} <:
       AbstractExplicitLayer{M, N}
    activation::AT
end

@doc raw"""
    VolumePreservingLowerLayer(dim)

Make an instance of `VolumePreservingLowerLayer` for a specific system dimension.

See the documentation for [`VolumePreservingFeedForwardLayer`](@ref).

# Examples

We apply the `VolumePreservingLowerLayer` with matrix:

```math
L = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}
```

to a vector:

```math
x = \begin{pmatrix} 1 \\ 1 \end{pmatrix}
```

```jldoctest
using GeometricMachineLearning

l = VolumePreservingLowerLayer(2, identity; use_bias=false)
A = StrictlyLowerTriangular([1,], 2)
ps = (weight = A, )
x = ones(eltype(A), 2)

l(x, ps)

# output

2-element Vector{Int64}:
 1
 2
```

# Arguments 

The constructor can be called with the optional arguments:
- `activation=tanh`: the activation function. 
- `use_bias::Bool=true` (keyword argument): specifies whether a bias should be used. 
"""
const VolumePreservingLowerLayer{M, N, UseBias, AT} = VolumePreservingFeedForwardLayer{
    M, N, UseBias, AT, :lower}

@doc raw"""
    VolumePreservingUpperLayer(dim)

Make an instance of `VolumePreservingUpperLayer` for a specific system dimension.

See the documentation for [`VolumePreservingFeedForwardLayer`](@ref).

# Examples

We apply the `VolumePreservingUpperLayer` with matrix:

```math
U = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}
```

to a vector:

```math
x = \begin{pmatrix} 1 \\ 1 \end{pmatrix}
```

```jldoctest
using GeometricMachineLearning

l = VolumePreservingUpperLayer(2, identity; use_bias=false)
A = StrictlyUpperTriangular([1,], 2)
ps = (weight = A, )
x = ones(eltype(A), 2)

l(x, ps)

# output

2-element Vector{Int64}:
 2
 1
```

# Arguments 

The constructor can be called with the optional arguments:
- `activation=tanh`: the activation function. 
- `use_bias::Bool=true` (keyword argument): specifies whether a bias should be used. 
"""
const VolumePreservingUpperLayer{M, N, UseBias, AT} = VolumePreservingFeedForwardLayer{
    M, N, UseBias, AT, :upper}

# The keyword's value is the type parameter, so the choice needs no branch.
function VolumePreservingLowerLayer(dim::Int, activation = tanh; use_bias::Bool = true)
    VolumePreservingFeedForwardLayer{dim, dim, use_bias, typeof(activation), :lower}(activation)
end

function VolumePreservingUpperLayer(dim::Int, activation = tanh; use_bias::Bool = true)
    VolumePreservingFeedForwardLayer{dim, dim, use_bias, typeof(activation), :upper}(activation)
end

# The strictly triangular leaf of the weight storage, which is the same vector for both triangles.
_strictly_triangular(::Val{:lower}, S, M) = StrictlyLowerTriangular(S, M)
_strictly_triangular(::Val{:upper}, S, M) = StrictlyUpperTriangular(S, M)

function parameterlength(
        ::VolumePreservingFeedForwardLayer{M, M, false}) where {M}
    M * (M - 1) ÷ 2
end
function parameterlength(
        ::VolumePreservingFeedForwardLayer{M, M, true}) where {M}
    M * (M - 1) ÷ 2 + M
end

function initialparameters(
        rng::AbstractRNG, init_weight::AbstractNeuralNetworks.Initializer,
        d::VolumePreservingFeedForwardLayer{M, M, UseBias, AT, Triangle},
        backend::Backend,
        ::Type{T}; init_bias = ZeroInitializer()) where {M, UseBias, AT, Triangle, T}
    S = KernelAbstractions.allocate(backend, T, M * (M - 1) ÷ 2)
    init_weight(rng, S)
    weight = _strictly_triangular(Val(Triangle), S, M)
    UseBias || return (weight = weight,)

    b = KernelAbstractions.allocate(backend, T, M)
    init_bias(rng, b)
    (weight = weight, bias = b)
end

function (d::VolumePreservingFeedForwardLayer{M, M, true})(x::AbstractArray, ps) where {M}
    x + d.activation.(_mul(ps.weight, x) .+ ps.bias)
end

function (d::VolumePreservingFeedForwardLayer{M, M, false})(x::AbstractArray, ps) where {M}
    x + d.activation.(_mul(ps.weight, x))
end
