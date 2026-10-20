@doc raw"""
    VolumePreservingFeedForwardLayer <: AbstractExplicitLayer

The layer type of which [`VolumePreservingLowerLayer`](@ref) and [`VolumePreservingUpperLayer`](@ref) are the two triangles. The layers do the following:

```math
x \mapsto \begin{cases} \sigma(Lx + b) & \text{where $L$ is }\mathtt{StrictlyLowerTriangular}, \\ \sigma(Ux + b) & \text{where $U$ is }\mathtt{StrictlyUpperTriangular}. \end{cases}
```

The functor can be applied to a vector, a matrix or a tensor. The special matrices are implemented as [`StrictlyLowerTriangular`](@extref GeometricOptimizers GeometricOptimizers.StrictlyLowerTriangular) and [`StrictlyUpperTriangular`](@extref GeometricOptimizers GeometricOptimizers.StrictlyUpperTriangular).
"""
struct VolumePreservingFeedForwardLayer{M, N, bias, AT, Triangle} <:
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
const VolumePreservingLowerLayer{M, N, bias, AT} = VolumePreservingFeedForwardLayer{
    M, N, bias, AT, :lower}

function VolumePreservingLowerLayer(sys_dim::Int, activation = tanh; use_bias::Bool = true)
    VolumePreservingLowerLayer{
        sys_dim, sys_dim, use_bias ? :bias : :no_bias, typeof(activation)}(
        activation)
end

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
const VolumePreservingUpperLayer{M, N, bias, AT} = VolumePreservingFeedForwardLayer{
    M, N, bias, AT, :upper}

function VolumePreservingUpperLayer(sys_dim::Int, activation = tanh; use_bias::Bool = true)
    VolumePreservingUpperLayer{
        sys_dim, sys_dim, use_bias ? :bias : :no_bias, typeof(activation)}(
        activation)
end

_strictly_triangular(::Val{:lower}) = StrictlyLowerTriangular
_strictly_triangular(::Val{:upper}) = StrictlyUpperTriangular

function parameterlength(::VolumePreservingFeedForwardLayer{M, M, :no_bias}) where {M}
    M * (M - 1) ÷ 2
end
function parameterlength(::VolumePreservingFeedForwardLayer{M, M, :bias}) where {M}
    M * (M - 1) ÷ 2 + M
end

function initialparameters(
        rng::AbstractRNG, init_weight::AbstractNeuralNetworks.Initializer,
        ::VolumePreservingFeedForwardLayer{M, M, bias, AT, Triangle}, backend::Backend,
        ::Type{T}; init_bias = ZeroInitializer()) where {M, bias, AT, Triangle, T}
    S = KernelAbstractions.allocate(backend, T, M * (M - 1) ÷ 2)
    init_weight(rng, S)
    weight = _strictly_triangular(Val(Triangle))(S, M)
    merge((weight = weight,), _bias_parameters(Val(bias), rng, init_bias, backend, T, M))
end

function (d::VolumePreservingFeedForwardLayer{M, M, :bias})(x::AbstractArray, ps) where {M}
    x + d.activation.(_mul(ps.weight, x) .+ ps.bias)
end

function (d::VolumePreservingFeedForwardLayer{M, M, :no_bias})(x::AbstractArray, ps) where {M}
    x + d.activation.(_mul(ps.weight, x))
end
