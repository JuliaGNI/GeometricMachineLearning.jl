@doc raw"""
    SympNetLayer <: AbstractExplicitLayer

Implements the various layers from the SympNet paper [jin2020sympnets](@cite).

This is a super type of [`GradientLayer`](@ref), [`ActivationLayer`](@ref) and [`LinearLayer`](@ref).

See the relevant docstrings of those layers for more information.
"""
abstract type SympNetLayer{M, N} <: AbstractExplicitLayer{M, N} end

@doc raw"""
    GradientLayer <: SympNetLayer

See the docstrings for the constructors [`GradientLayerQ`](@ref) and [`GradientLayerP`](@ref).
""" 
struct GradientLayer{M, N, TA, C} <: SympNetLayer{M, N}
        second_dim::Integer
        activation::TA
end

@doc raw"""
    GradientLayerQ(n, upscaling_dimension, activation)

Make an instance of a gradient-``q`` layer.

The gradient layer that changes the ``q`` component. It is of the form: 

```math
\begin{bmatrix}
        \mathbb{I} & \nabla{}V \\ \mathbb{O} & \mathbb{I} 
\end{bmatrix},
```

with ``V(p) = \sum_{i=1}^Ma_i\Sigma(\sum_jk_{ij}p_j+b_i)``, where ``\mathtt{activation} \equiv \Sigma`` is the antiderivative of the activation function ``\sigma`` (one-layer neural network). We refer to ``M`` as the *upscaling dimension*. 

Such layers are by construction symplectic.
"""
const GradientLayerQ{M, N, TA} = GradientLayer{M, N, TA, :Q}

@doc raw"""
    GradientLayerP(n, upscaling_dimension, activation)

Make an instance of a gradient-``p`` layer.

The gradient layer that changes the ``p`` component. It is of the form: 

```math
\begin{bmatrix}
        \mathbb{I} & \mathbb{O} \\ \nabla{}V & \mathbb{I} 
\end{bmatrix},
```

with ``V(p) = \sum_{i=1}^Ma_i\Sigma(\sum_jk_{ij}q_j+b_i)``, where ``\mathtt{activation} \equiv \Sigma`` is the antiderivative of the activation function ``\sigma`` (one-layer neural network). We refer to ``M`` as the *upscaling dimension*. 

Such layers are by construction symplectic.
"""
const GradientLayerP{M, N, TA} = GradientLayer{M, N, TA, :P}

@doc raw"""
    LinearLayer <: SympNetLayer

See the constructors [`LinearLayerQ`](@ref) and [`LinearLayerP`](@ref).

# Implementation

`LinearLayer` uses the custom matrix [`SymmetricMatrix`](@ref) for its weight. 
"""
struct LinearLayer{M, N, C} <: SympNetLayer{M, N}
end

@doc raw"""
    LinearLayerQ(n)

Make a linear layer of dimension ``n\times{}n`` that only changes the ``q`` component.

This is equivalent to a left multiplication by the matrix:
```math
\begin{pmatrix}
\mathbb{I} & A \\ 
\mathbb{O} & \mathbb{I}
\end{pmatrix}, 
```
where ``A`` is a [`SymmetricMatrix`](@ref).
"""
const LinearLayerQ{M, N, TA} = LinearLayer{M, N, :Q}

@doc raw"""
    LinearLayerP(n)

Make a linear layer of dimension ``n\times{}n`` that only changes the ``p`` component.

This is equivalent to a left multiplication by the matrix:
```math
\begin{pmatrix}
\mathbb{I} & \mathbb{O} \\ 
A & \mathbb{I}
\end{pmatrix}, 
```
where ``A`` is a [`SymmetricMatrix`](@ref).
"""
const LinearLayerP{M, N, TA} = LinearLayer{M, N, :P}

@doc raw"""
    ActivationLayer <: SympNetLayer

See the constructors [`ActivationLayerQ`](@ref) and [`ActivationLayerP`](@ref).
"""
struct  ActivationLayer{M, N, TA, C} <: SympNetLayer{M, N}
        activation::TA
end

@doc raw"""
    ActivationLayerQ(n, σ)

Make an activation layer of size `n` and with activation `σ` that only changes the ``q`` component. 

Performs:

```math
\begin{pmatrix}
        q \\ p
\end{pmatrix} \mapsto 
\begin{pmatrix}
        q + \mathrm{diag}(a)\sigma(p) \\ p
\end{pmatrix}.
```

This can be recovered from [`GradientLayerQ`](@ref) by setting ``M`` equal to `n`, ``K`` equal to ``\mathbb{I}`` and ``b`` equal to zero.
"""
const ActivationLayerQ{M, N, TA} = ActivationLayer{M, N, TA, :Q}

@doc raw"""
    ActivationLayerP(n, σ)

Make an activation layer of size `n` and with activation `σ` that only changes the ``p`` component. 

Performs:

```math
\begin{pmatrix}
        q \\ p
\end{pmatrix} \mapsto 
\begin{pmatrix}
        q \\ p + \mathrm{diag}(a)\sigma(q)
\end{pmatrix}.
```

This can be recovered from [`GradientLayerP`](@ref) by setting ``M`` equal to `n`, ``K`` equal to ``\mathbb{I}`` and ``b`` equal to zero.
"""
const ActivationLayerP{M, N, TA} = ActivationLayer{M, N, TA, :P}

function GradientLayerQ(M, upscaling_dimension, activation)
        GradientLayer{M, M, typeof(activation), :Q}(upscaling_dimension, activation)
end

function GradientLayerP(M, upscaling_dimension, activation)
        GradientLayer{M, M, typeof(activation), :P}(upscaling_dimension, activation)
end

function ActivationLayerQ(M, activation)
        ActivationLayerQ{M, M, typeof(activation)}(activation)
end

function ActivationLayerP(M, activation)
        ActivationLayerP{M, M, typeof(activation)}(activation)
end

function LinearLayerQ(M)
        LinearLayerQ{M, M}()
end

function LinearLayerP(M)
        LinearLayerP{M, M}()
end

function Gradient(dim::Int, dim2::Int=dim, activation = identity; full_grad::Bool=true, change_q::Bool=true, allow_fast_activation::Bool=true)
        @warn "You are calling the old constructor. This will be deprecated."

        activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
        
        iseven(dim) && iseven(dim2) || error("Dimensions must be even!")
        dim2 ≥ dim || error("Second dimension should be bigger than the first!")

        if change_q
                return full_grad ? GradientLayerQ{dim, dim, typeof(activation)}(dim2, activation) : ActivationLayerQ{dim, dim, typeof(activation)}(activation)
        else
                return full_grad ? GradientLayerP{dim, dim, typeof(activation)}(dim2, activation) : ActivationLayerP{dim, dim, typeof(activation)}(activation)
        end
end

function initialparameters(rng::AbstractRNG, init_weight::AbstractNeuralNetworks.Initializer, d::GradientLayer{M, M}, backend::Backend, ::Type{T}; init_bias = ZeroInitializer(), init_scale = GlorotUniform()) where {M, T}
        K = KernelAbstractions.allocate(backend, T, d.second_dim÷2, M÷2)
        b = KernelAbstractions.allocate(backend, T, d.second_dim÷2)
        a = KernelAbstractions.allocate(backend, T, d.second_dim÷2)
        init_weight(rng, K)
        init_bias(rng, b)
        init_scale(rng, a)
        return (weight=K, bias=b, scale=a)
end

function initialparameters(rng::AbstractRNG, init_scale::AbstractNeuralNetworks.Initializer, ::ActivationLayer{M, M}, backend::Backend, ::Type{T}) where {M, T}
        a = KernelAbstractions.zeros(backend, T, M ÷ 2)
        init_scale(rng, a)
        return (scale = a,)
end

function initialparameters(rng::AbstractRNG, init_weight::AbstractNeuralNetworks.Initializer, ::LinearLayer{M, M}, backend::Backend, ::Type{T}) where {M, T}
        S = KernelAbstractions.allocate(backend, T, (M ÷ 2) * (M ÷ 2 + 1) ÷ 2)
        init_weight(rng, S)
        (weight=SymmetricMatrix(S, M ÷ 2), )
end

function parameterlength(d::GradientLayer{M, M}) where {M}
        d.second_dim ÷ 2 * (M ÷ 2 + 2)
end

function parameterlength(::ActivationLayer{M, M}) where {M}
        M ÷ 2
end

function parameterlength(::LinearLayer{M, M}) where {M}
        (M ÷ 2) * (M ÷ 2 + 1) ÷ 2
end

# Multiplies a matrix with a vector, a matrix or a tensor.
custom_mat_mul(weight::AbstractMatrix, x::AbstractVecOrMat) = weight*x 
function custom_mat_mul(weight::AbstractMatrix, x::AbstractArray{T, 3}) where T 
        mat_tensor_mul(weight, x)
end

function custom_vec_mul(scale::AbstractVector, x::AbstractVecOrMat) 
        scale .* x 
end
function custom_vec_mul(scale::AbstractVector{T}, x::AbstractArray{T, 3}) where T 
        vec_tensor_mul(scale, x)
end

@inline function (d::ActivationLayerQ{M, M})(x::NamedTuple, ps) where {M}
        size(x.q, 1) == M÷2 || error("Dimension mismatch.")
        return (q = x.q + custom_vec_mul(ps.scale, d.activation.(x.p)), p = x.p)
end

@inline function (d::ActivationLayerP{M, M})(x::NamedTuple, ps) where {M}
        size(x.q, 1) == M÷2 || error("Dimension mismatch.")
        return (q = x.q, p = x.p + custom_vec_mul(ps.scale, d.activation.(x.q)))
end


@inline function (d::GradientLayerQ{M, M})(x::NamedTuple, ps) where {M}
        size(x.q, 1) == M÷2 || error("Dimension mismatch.")
        (q = x.q + custom_mat_mul(ps.weight', (custom_vec_mul(ps.scale, d.activation.(custom_mat_mul(ps.weight, x.p) .+ ps.bias)))), p = x.p)     
end


@inline function(d::GradientLayerP{M, M})(x::NamedTuple, ps) where {M}
        size(x.q, 1) == M÷2 || error("Dimension mismatch.")
        (q = x.q, p = x.p + custom_mat_mul(ps.weight', (custom_vec_mul(ps.scale, d.activation.(custom_mat_mul(ps.weight, x.q) .+ ps.bias)))))
end


@inline function(d::LinearLayerQ{M, M})(x::NamedTuple, ps) where {M}
        size(x.q, 1) == M÷2 || error("Dimension mismatch.")
        (q = x.q + custom_mat_mul(ps.weight, x.p), p = x.p)
end


@inline function(d::LinearLayerP{M, M})(x::NamedTuple, ps) where {M}
        size(x.q, 1) == M÷2 || error("Dimension mismatch.")
        (q = x.q, p = x.p + custom_mat_mul(ps.weight, x.q))
end


# @doc raw"""
# This function is used in the wrappers where the input to the SympNet layers is not a `NamedTuple` (as it should be) but an `AbstractArray`.
# 
# It converts the Array to a `NamedTuple` (via `assign_q_and_p`), then calls the SympNet routine(s) and converts back to an `AbstractArray` (with `vcat`).
# """
function apply_layer_to_nt_and_return_array(x::AbstractArray, d::AbstractExplicitLayer, ps::NamedTuple)
        N2 = size(x, 1)÷2
        qp = assign_q_and_p(x, N2)
        output = d(qp, ps)
        return vcat(output.q, output.p)
end

# @doc raw"""
# This is called when a [`SympNetLayer`](@ref) is applied to a `NamedTuple`. 
# 
# It calls [`GeometricMachineLearning.apply_layer_to_nt_and_return_array`](@ref).
# """
@inline function (d::SympNetLayer)(x::AbstractArray, ps::NamedTuple)
        apply_layer_to_nt_and_return_array(x, d, ps)
end