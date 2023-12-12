@doc raw"""
Implements the various layers from the SympNet paper: (https://www.sciencedirect.com/science/article/abs/pii/S0893608020303063). This is a super type of `Gradient`, `Activation` and `Linear`.

For the linear layer, the activation and the bias are left out, and for the activation layer $K$ and $b$ are left out!
"""
abstract type SympNetLayer{M, N} <: AbstractExplicitLayer{M, N} end

@doc raw"""
`GradientLayer` is the `struct` corresponding to the constructors `GradientLayerQ` and `GradientLayerP`. See those for more information.
""" 
struct GradientLayer{M, N, TA, C} <: SympNetLayer{M, N}
        second_dim::Integer
        activation::TA
end

const GradientLayerQ{M, N, TA} = GradientLayer{M, N, TA, :Q}
const GradientLayerP{M, N, TA} = GradientLayer{M, N, TA, :P}

@doc raw"""
`LinearLayer` is the `struct` corresponding to the constructors `LinearLayerQ` and `LinearLayerP`. See those for more information.
"""
struct LinearLayer{M, N, C} <: SympNetLayer{M, N}
end

const LinearLayerQ{M, N, TA} = LinearLayer{M, N, :Q}
const LinearLayerP{M, N, TA} = LinearLayer{M, N, :P}

@doc raw"""
`ActivationLayer` is the `struct` corresponding to the constructors `ActivationLayerQ` and `ActivationLayerP`. See those for more information.
"""
struct  ActivationLayer{M, N, TA, C} <: SympNetLayer{M, N}
        activation::TA
end

const ActivationLayerQ{M, N, TA} = ActivationLayer{M, N, TA, :Q}
const ActivationLayerP{M, N, TA} = ActivationLayer{M, N, TA, :P}

@doc raw"""
The gradient layer that changes the $q$ component. It is of the form: 

```math
\begin{bmatrix}
        \mathbb{I} & \nabla{}V \\ \mathbb{O} & \mathbb{I} 
\end{bmatrix},
```

with $V(p) = \sum_{i=1}^Ma_i\Sigma(\sum_jk_{ij}p_j+b_i)$, where $\Sigma$ is the antiderivative of the activation function $\sigma$ (one-layer neural network). We refer to $M$ as the *upscaling dimension*. Such layers are by construction symplectic.
"""
function GradientLayerQ(M, upscaling_dimension, activation)
        GradientLayer{M, M, typeof(activation), :Q}(upscaling_dimension, activation)
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
function GradientLayerP(M, upscaling_dimension, activation)
        GradientLayer{M, M, typeof(activation), :P}(upscaling_dimension, activation)
end

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
function ActivationLayerQ(M, activation)
        ActivationLayerQ{M, M, typeof(activation)}(activation)
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
function ActivationLayerP(M, activation)
        ActivationLayerP{M, M, typeof(activation)}(activation)
end


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
function LinearLayerQ(M)
        LinearLayerQ{M, M}()
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
function LinearLayerP(M)
        LinearLayerP{M, M}()
end

@doc raw"""
This is an old constructor and will be depricated. For `change_q=true` it is equivalent to `GradientLayerQ`; for `change_q=false` it is equivalent to `GradientLayerP`.

If `full_grad=false` then `ActivationLayer` is called
"""
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

function initialparameters(backend::Backend, ::Type{T}, d::GradientLayer{M, M}; rng::AbstractRNG = Random.default_rng(), init_weight = GlorotUniform(), init_bias = ZeroInitializer(), init_scale = GlorotUniform()) where {M, T}
        K = KernelAbstractions.allocate(backend, T, d.second_dim÷2, M÷2)
        b = KernelAbstractions.allocate(backend, T, d.second_dim÷2)
        a = KernelAbstractions.allocate(backend, T, d.second_dim÷2)
        init_weight(rng, K)
        init_bias(rng, b)
        init_scale(rng, a)
        return (weight=K, bias=b, scale=a)
end

function initialparameters(backend::Backend, ::Type{T}, ::ActivationLayer{M, M}; rng::AbstractRNG = Random.default_rng(), init_scale = GlorotUniform()) where {M, T}
        a = KernelAbstractions.zeros(backend, T, M÷2)
        init_scale(rng, a)
        return (scale = a,)
end

function initialparameters(backend::Backend, ::Type{T}, ::LinearLayer{M, M}; rng::AbstractRNG = Random.default_rng(), init_weight = GlorotUniform()) where {M, T}
        S = KernelAbstractions.allocate(backend, T, (M÷2)*(M÷2+1)÷2)
        init_weight(rng, S)
        (weight=SymmetricMatrix(S, M÷2), )
end

function parameterlength(d::GradientLayer{M, M}) where {M}
        d.second_dim÷2 * (M÷2 + 2)
end

function parameterlength(::ActivationLayer{M, M}) where {M}
        M÷2
end

function parameterlength(::LinearLayer{M, M}) where {M}
        (M÷2)*(M÷2+1)÷2
end

@doc raw"""
Multiplies a matrix with a vector, a matrix or a tensor.
"""
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


@doc raw"""
This function is used in the wrappers where the input to the SympNet layers is not a `NamedTuple` (as it should be) but an `AbstractArray`.

It converts the Array to a `NamedTuple` (via `assign_q_and_p`), then calls the SympNet routine(s) and converts back to an `AbstractArray` (with `vcat`).
"""
function apply_layer_to_nt_and_return_array(x::AbstractArray, d::SympNetLayer{M, M}, ps) where {M}
        N2 = size(x, 1)÷2
        qp = assign_q_and_p(x, N2)
        output = d(qp, ps)
        return vcat(output.q, output.p)
end

@doc raw"""
This is called when a SympnetLayer is applied to a `NamedTuple`. It calls `apply_layer_to_nt_and_return_array`.
"""
@inline function (d::SympNetLayer)(x::AbstractArray, ps)
        apply_layer_to_nt_and_return_array(x, d, ps)
end