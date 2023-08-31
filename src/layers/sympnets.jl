@doc raw"""
Implements the various layers from the SympNet paper: (https://www.sciencedirect.com/science/article/abs/pii/S0893608020303063). 
Its components are of the form: 

```math
\begin{pmatrix}
        I & \nabla{}V \\ 0 & I 
\end{pmatrix},
```

with $V(p) = \sum_ia_i\Sigma(\sum_jk_{ij}p_j+b_i)$, where $\Sigma$ is the antiderivative of the activation function $\sigma$ (one-layer neural network). Such layers are by construction symplectic.

For the linear layer, the activation and the bias are left out, and for the activation layer K and b are left out!
"""
abstract type SympNetLayer{M, N} <: AbstractExplicitLayer{M, N} end

abstract type Gradient{M, N, TA} <: SympNetLayer{M, N} end

struct GradientQ{M, N, TA} <: Gradient{M, N, TA}
        second_dim::Integer
        activation::TA
end
function GradientQ(M, second_dim, activation)
        GradientQ{M, M, typeof(activation)}(second_dim, activation)
end

struct GradientP{M, N, TA} <: Gradient{M, N, TA}
        second_dim::Integer
        activation::TA
end
function GradientP(M, second_dim, activation)
        GradientP{M, M, typeof(activation)}(second_dim, activation)
end

abstract type Activation{M, N, TA} <: SympNetLayer{M, N} end

struct ActivationQ{M, N, TA} <: Activation{M, N, TA} 
        activation::TA
end
function ActivationQ(M, activation)
        ActivationQ{M, M, typeof(activation)}(activation)
end

struct ActivationP{M, N, TA} <: Activation{M, N, TA} 
        activation::TA
end
function ActivationP(M, activation)
        ActivationP{M, M, typeof(activation)}(activation)
end

abstract type LinearSympNetLayer{M, N} <: SympNetLayer{M, N} end

struct LinearQ{M, N} <: LinearSympNetLayer{M, N} 
end 
function LinearQ(M)
        LinearQ{M, M}()
end

struct LinearP{M, N} <: LinearSympNetLayer{M, N}
end 
function LinearP(M)
        LinearP{M, M}()
end

# check: input is even; make dim2 an optional argument for full_grad=false
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


@inline function (d::ActivationQ{M, M})(x::AbstractArray, ps) where {M}
        size(x)[1] == M || error("Dimension mismatch.")
        N2 = M÷2
        q, p = assign_q_and_p(x, N2)
        return vcat(q + custom_vec_mul(ps.scale, d.activation.(p)), p)
end

@inline function (d::ActivationP{M, M})(x::AbstractArray, ps) where {M}
        size(x)[1] == M || error("Dimension mismatch.")
        N2 = M÷2 
        q, p = assign_q_and_p(x, N2)
        return vcat(q, p + custom_vec_mul(ps.scale, d.activation.(q)))
end

@inline function (d::GradientQ{M, M})(x::AbstractArray, ps) where {M}
        size(x)[1] == M || error("Dimension mismatch.")
        N2 = M÷2 
        q, p = assign_q_and_p(x, N2)
        return vcat(q + custom_mat_mul(ps.weight', (custom_vec_mul(ps.scale, d.activation.(custom_mat_mul(ps.weight, p) .+ ps.bias)))), p)
end

@inline function(d::GradientP{M, M})(x::AbstractArray, ps) where {M}
        size(x)[1] == M || error("Dimension mismatch.")
        N2 = M÷2 
        q, p = assign_q_and_p(x, N2)
        return vcat(q, p + custom_mat_mul(ps.weight', (custom_vec_mul(ps.scale, d.activation.(custom_mat_mul(ps.weight, q) .+ ps.bias)))))
end

@inline function (d::LinearQ{M, M})(x::AbstractArray, ps) where {M}
        size(x)[1] == M || error("Dimension mismatch.")
        N2 = M÷2 
        q, p = assign_q_and_p(x, N2)
        vcat(q + custom_mat_mul(ps.weight,p), p)
end

@inline function (d::LinearP{M, M})(x::AbstractArray, ps) where {M}
        size(x)[1] == M || error("Dimension mismatch.")
        N2 = M÷2 
        q, p = assign_q_and_p(x, N2)
        vcat(q, p + custom_mat_mul(ps.weight,q))
end