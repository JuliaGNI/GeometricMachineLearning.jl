@doc raw"""
Implements the various layers from the SympNet paper: (https://www.sciencedirect.com/science/article/abs/pii/S0893608020303063). This is a super type of `Gradient`, `Activation` and `Linear`.

For the linear layer, the activation and the bias are left out, and for the activation layer $K$ and $b$ are left out!
"""
abstract type SympNetLayer{M, N} <: AbstractExplicitLayer{M, N} end

@doc raw"""
Super type of `GradientQ` and `GradientP`.
""" 
abstract type Gradient{M, N, TA} <: SympNetLayer{M, N} end

@doc raw"""
The gradient layer that changes the $q$ component. It is of the form: 

```math
\begin{bmatrix}
        \mathbb{I} & \nabla{}V \\ \mathbb{O} & \mathbb{I} 
\end{bmatrix},
```

with $V(p) = \sum_{i=1}^Ma_i\Sigma(\sum_jk_{ij}p_j+b_i)$, where $\Sigma$ is the antiderivative of the activation function $\sigma$ (one-layer neural network). We refer to $M$ as the *upscaling dimension*. Such layers are by construction symplectic.
"""
struct GradientQ{M, N, TA} <: Gradient{M, N, TA}
        second_dim::Integer
        activation::TA
end

function GradientQ(M, upscaling_dimension, activation)
        GradientQ{M, M, typeof(activation)}(upscaling_dimension, activation)
end

@doc raw"""
The gradient layer that changes the $q$ component. It is of the form: 

```math
\begin{bmatrix}
        \mathbb{I} & \mathbb{O} \\ \nabla{}V & \mathbb{I} 
\end{bmatrix},
```

with $V(p) = \sum_{i=1}^Ma_i\Sigma(\sum_jk_{ij}p_j+b_i)$, where $\Sigma$ is the antiderivative of the activation function $\sigma$ (one-layer neural network). We refer to $M$ as the *upscaling dimension*. Such layers are by construction symplectic.
"""
struct GradientP{M, N, TA} <: Gradient{M, N, TA}
        second_dim::Integer
        activation::TA
end

function GradientP(M, upscaling_dimension, activation)
        GradientP{M, M, typeof(activation)}(upscaling_dimension, activation)
end

@doc raw"""
Super type of `ActivationQ` and `ActivationP`.
"""
abstract type Activation{M, N, TA} <: SympNetLayer{M, N} end

@doc raw"""
Performs:

```math
\begin{pmatrix}
        q \\ p
\end{pmatrix} \mapsto 
\begin{pmatrix}
        q + \mathrm{diag}(a)\sigma(p) \\ p
\end{pmatrix}.
```

"""
struct ActivationQ{M, N, TA} <: Activation{M, N, TA} 
        activation::TA
end
function ActivationQ(M, activation)
        ActivationQ{M, M, typeof(activation)}(activation)
end

@doc raw"""
Performs:

```math
\begin{pmatrix}
        q \\ p
\end{pmatrix} \mapsto 
\begin{pmatrix}
        q \\ p + \mathrm{diag}(a)\sigma(q)
\end{pmatrix}.
```

"""
struct ActivationP{M, N, TA} <: Activation{M, N, TA} 
        activation::TA
end
function ActivationP(M, activation)
        ActivationP{M, M, typeof(activation)}(activation)
end

@doc raw"""
Super type of `LinearQ` and `LinearP`.
"""
abstract type LinearSympNetLayer{M, N} <: SympNetLayer{M, N} end

@doc raw"""
Equivalent to a left multiplication by the matrix:
```math
\begin{pmatrix}
\mathbb{I} & B \\ 
\mathbb{O} & \mathbb{I}
\end{pmatrix}, 
```
where $B$ is a symmetric matrix.
"""
struct LinearQ{M, N} <: LinearSympNetLayer{M, N} 
end 

function LinearQ(M)
        LinearQ{M, M}()
end

@doc raw"""
Equivalent to a left multiplication by the matrix:
```math
\begin{pmatrix}
\mathbb{I} & \mathbb{O} \\ 
B & \mathbb{I}
\end{pmatrix}, 
```
where $B$ is a symmetric matrix.
"""
struct LinearP{M, N} <: LinearSympNetLayer{M, N}
end 

function LinearP(M)
        LinearP{M, M}()
end

"""
This is an old constructor and will be depricated. For `change_q=true` it is equivalent to `GradientQ`; for `change_q=false` it is equivalent to `GradientP`.
"""
function Gradient(dim::Int, dim2::Int=dim, activation = identity; full_grad::Bool=true, change_q::Bool=true, allow_fast_activation::Bool=true)
        @warn "You are calling the old constructor. This will be deprecated."

        activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
        
        iseven(dim) && iseven(dim2) || error("Dimensions must be even!")
        dim2 ≥ dim || error("Second dimension should be bigger than the first!")

        if change_q
                return full_grad ? GradientQ{dim, dim, typeof(activation)}(dim2, activation) : ActivationQ{dim, dim, typeof(activation)}(activation)
        else
                return full_grad ? GradientP{dim, dim, typeof(activation)}(dim2, activation) : ActivationP{dim, dim, typeof(activation)}(activation)
        end
end

function initialparameters(backend::Backend, ::Type{T}, d::Gradient{M, M}; rng::AbstractRNG = Random.default_rng(), init_weight = GlorotUniform(), init_bias = ZeroInitializer(), init_scale = GlorotUniform()) where {M, T}
        K = KernelAbstractions.allocate(backend, T, d.second_dim÷2, M÷2)
        b = KernelAbstractions.allocate(backend, T, d.second_dim÷2)
        a = KernelAbstractions.allocate(backend, T, d.second_dim÷2)
        init_weight(rng, K)
        init_bias(rng, b)
        init_scale(rng, a)
        return (weight=K, bias=b, scale=a)
end

function initialparameters(backend::Backend, ::Type{T}, ::Activation{M, M}; rng::AbstractRNG = Random.default_rng(), init_scale = GlorotUniform()) where {M, T}
        a = KernelAbstractions.zeros(backend, T, M÷2)
        init_scale(rng, a)
        return (scale = a,)
end

function initialparameters(backend::Backend, ::Type{T}, ::LinearSympNetLayer{M, M}; rng::AbstractRNG = Random.default_rng(), init_weight = GlorotUniform()) where {M, T}
        S = KernelAbstractions.allocate(backend, T, (M÷2)*(M÷2+1)÷2)
        init_weight(rng, S)
        (weight=SymmetricMatrix(S, M÷2), )
end

function parameterlength(d::Gradient{M, M}) where {M}
        d.second_dim÷2 * (M÷2 + 2)
end

function parameterlength(::Activation{M, M}) where {M}
        M÷2
end

function parameterlength(::Linear{M, M}) where {M}
        (M÷2)*(M÷2+1)÷2
end

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

@doc raw"""
This function is used in the wrappers where the input to the SympNet layers is not a `NamedTuple` (as it should be) but an `Array`.
"""
function apply_layer_to_nt_and_return_array(d::SympNetLayer{M, M}, ps) where {M}
        q, p = assign_q_and_p(x, N2)
        output = d((q = q, p = p), ps)
        return vcat(output.q, output.p)
end

@inline function (d::Activation{M, M})(x::NamedTuple, ps) where {M}
        size(x.q, 1) == M÷2 || error("Dimension mismatch.")
        return (q = x.q + custom_vec_mul(ps.scale, d.activation.(x.p)), p = x.p)
end

@inline function (d::ActivationQ)(x::AbstractArray, ps)
        apply_layer_to_nt_and_return_array(d, ps)
end

@inline function (d::ActivationP{M, M})(x::NamedTuple, ps) where {M}
        size(x.q, 1) == M÷2 || error("Dimension mismatch.")
        return (q = x.q, p = x.p + custom_vec_mul(ps.scale, d.activation.(x.q)))
end

@inline function (d::ActivationP)(x::AbstractArray, ps)
        apply_layer_to_nt_and_return_array(d, ps)
end

@inline function (d::GradientQ{M, M})(x::NamedTuple, ps) where {M}
        size(x.q, 1) == M÷2 || error("Dimension mismatch.")
        (q = x.q + custom_mat_mul(ps.weight', (custom_vec_mul(ps.scale, d.activation.(custom_mat_mul(ps.weight, x.p) .+ ps.bias)))), p = x.p)     
end

@inline function (d::GradientQ)(x::AbstractArray, ps) 
        apply_layer_to_nt_and_return_array(d, ps)
end

@inline function(d::GradientP{M, M})(x::NamedTuple, ps) where {M}
        size(x.q, 1) == M÷2 || error("Dimension mismatch.")
        (q = x.q, p = x.p + custom_mat_mul(ps.weight', (custom_vec_mul(ps.scale, d.activation.(custom_mat_mul(ps.weight, x.q) .+ ps.bias)))))
end

@inline function(d::GradientP{M, M})(x::AbstractArray, ps) where {M}
        apply_layer_to_nt_and_return_array(d, ps)
end

@inline function(d::LinearQ{M, M})(x::NamedTuple, ps) where {M}
        size(x.q, 1) == M÷2 || error("Dimension mismatch.")
        (q = x.q + custom_mat_mul(ps.weight, x.p), p = x.p)
end

@inline function (d::LinearQ)(x::AbstractArray, ps)
        apply_layer_to_nt_and_return_array(d, ps)
end

@inline function(d::LinearP{M, M})(x::NamedTuple, ps) where {M}
        size(x.q, 1) == M÷2 || error("Dimension mismatch.")
        (q = x.q, p = x.p + custom_mat_mul(ps.weight, x.q))
end

@inline function (d::LinearP)(x::AbstractArray, ps)
        apply_layer_to_nt_and_return_array(d, ps)
end