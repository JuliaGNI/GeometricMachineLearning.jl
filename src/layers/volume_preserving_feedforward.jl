@doc raw"""
    VolumePreservingFeedForwardLayer <: AbstractExplicitLayer

Super-type of [`VolumePreservingLowerLayer`](@ref) and [`VolumePreservingUpperLayer`](@ref). The layers do the following: 

```math
x \mapsto \begin{cases} \sigma(Lx + b) & \text{where $L$ is }\mathtt{LowerTriangular}, \\ \sigma(Ux + b) & \text{where $U$ is }\mathtt{UpperTriangular}. \end{cases}
```

The functor can be applied to a vector, a matrix or a tensor. The special matrices are implemented as [`LowerTriangular`](@ref) and [`UpperTriangular`](@ref).
"""
abstract type VolumePreservingFeedForwardLayer{M, N, bias} <: AbstractExplicitLayer{M, N} end 

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
A = LowerTriangular([1,], 2)
ps = (weight = A, )
x = ones(eltype(A), 2)

l(x, ps)

# output

2×1 Matrix{Int64}:
 1
 2
```

# Arguments 

The constructor can be called with the optional arguments:
- `activation=tanh`: the activation function. 
- `use_bias::Bool=true` (keyword argument): specifies whether a bias should be used. 
"""
struct VolumePreservingLowerLayer{M, N, bias, AT} <: VolumePreservingFeedForwardLayer{M, N, bias}
    activation::AT

    function VolumePreservingLowerLayer(sys_dim::Int, activation=tanh; use_bias::Bool=true)
        if use_bias
            return new{sys_dim, sys_dim, :bias, typeof(activation)}(activation)
        else 
            return new{sys_dim, sys_dim, :no_bias, typeof(activation)}(activation)
        end
    end
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
A = UpperTriangular([1,], 2)
ps = (weight = A, )
x = ones(eltype(A), 2)

l(x, ps)

# output

2×1 Matrix{Int64}:
 2
 1
```

# Arguments 

The constructor can be called with the optional arguments:
- `activation=tanh`: the activation function. 
- `use_bias::Bool=true` (keyword argument): specifies whether a bias should be used. 
"""
struct VolumePreservingUpperLayer{M, N, bias, AT} <: VolumePreservingFeedForwardLayer{M, N, bias}
    activation::AT

    function VolumePreservingUpperLayer(sys_dim::Int, activation=tanh; use_bias::Bool=true)
        if use_bias
            return new{sys_dim, sys_dim, :bias, typeof(activation)}(activation)
        else 
            return new{sys_dim, sys_dim, :no_bias, typeof(activation)}(activation)
        end
    end
end 

parameterlength(::VolumePreservingFeedForwardLayer{M, M, :no_bias}) where M = M * (M - 1) ÷ 2 
parameterlength(::VolumePreservingFeedForwardLayer{M, M, :bias}) where M = M * (M - 1) ÷ 2 + M

function initialparameters(rng::AbstractRNG, init_weight!::AbstractNeuralNetworks.Initializer, d::VolumePreservingLowerLayer{M, M, :bias}, backend::Backend, ::Type{T}; init_bias! = ZeroInitializer()) where {M, T}
    S = KernelAbstractions.allocate(backend, T, parameterlength(d) - M)
    b = KernelAbstractions.allocate(backend, T, M)
    init_weight!(rng, S)
    init_bias!(rng, b)

    (weight = LowerTriangular(S, M), bias = b)
end 

function initialparameters(rng::AbstractRNG, init_weight!::AbstractNeuralNetworks.Initializer, d::VolumePreservingUpperLayer{M, M, :bias}, backend::Backend, ::Type{T}; init_bias! = ZeroInitializer()) where {M, T}
    S = KernelAbstractions.allocate(backend, T, parameterlength(d) - M)
    b = KernelAbstractions.allocate(backend, T, M)
    init_weight!(rng, S)
    init_bias!(rng, b)
    
    (weight = UpperTriangular(S, M), bias = b)
end 

function initialparameters(rng::AbstractRNG, init_weight!::AbstractNeuralNetworks.Initializer, d::VolumePreservingLowerLayer{M, M, :no_bias}, backend::Backend, ::Type{T}; init_bias! = ZeroInitializer()) where {M, T}
    S = KernelAbstractions.allocate(backend, T, parameterlength(d))
    init_weight!(rng, S)

    (weight = LowerTriangular(S, M), )
end 

function initialparameters(rng::AbstractRNG, init_weight!::AbstractNeuralNetworks.Initializer, d::VolumePreservingUpperLayer{M, M, :no_bias}, backend::Backend, ::Type{T}; init_bias! = ZeroInitializer()) where {M, T}
    S = KernelAbstractions.allocate(backend, T, parameterlength(d))
    init_weight!(rng, S)
    
    (weight = UpperTriangular(S, M), )
end 

function (d::VolumePreservingFeedForwardLayer{M, M, :bias})(x::AbstractArray{T, 3}, ps) where {T, M}
    x + d.activation.(mat_tensor_mul(ps.weight, x) .+ ps.bias)
end

function (d::VolumePreservingFeedForwardLayer{M, M, :no_bias})(x::AbstractArray{T, 3}, ps) where {T, M}
    x + d.activation.(mat_tensor_mul(ps.weight, x))
end

function (d::VolumePreservingFeedForwardLayer{M, M, :bias})(x::AbstractMatrix{T}, ps) where {T, M} 
    x + d.activation.(ps.weight * x .+ ps.bias)
end

function (d::VolumePreservingFeedForwardLayer{M, M, :bias})(x::AbstractVector{T}, ps) where {T, M} 
    x + d.activation.(ps.weight * x + ps.bias)
end

function (d::VolumePreservingFeedForwardLayer{M, M, :no_bias})(x::AbstractVecOrMat{T}, ps) where {T, M} 
    x + d.activation.(ps.weight * x)
end