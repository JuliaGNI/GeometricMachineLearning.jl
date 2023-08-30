@doc raw"""
The gradient layer from the SympNet paper (https://www.sciencedirect.com/science/article/abs/pii/S0893608020303063). 
Its components are of the form: 
$$
\begin{pmatrix}
        I & \nabla{}V \\ 0 & I 
\end{pmatrix},
$$
with $V(p) = \sum_ia_i\Sigma(\sum_jk_{ij}p_j+b_i)$, where $\Sigma$ is the antiderivative of the activation function $\sigma$. Such layers are by construction symplectic.

TODO: Implement tensor version! Implement tests!
"""
abstract type SympNet{M, N} <: AbstractExplicitLayer{M, N} end

abstract type Gradient{M, N, TA} <: SympNet{M, N} end

struct GradientQ{M, N, TA} <: Gradient{M, N, TA}
        second_dim::Integer
        activation::TA
end

struct GradientP{M, N, TA} <: Gradient{M, N, TA}
        second_dim::Integer
        activation::TA
end

abstract type Activation{M, N, TA} <: SympNet{M, N} end

struct ActivationQ{M, N, TA} <: Activation{M, N, TA} 
        activation::TA
end

struct ActivationP{M, N, TA} <: Activation{M, N, TA} 
        activation::TA
end

abstract type Linear{M, N} <: SympNet{M, N} end

struct LinearQ{M, N} <: SympNet{M, N} 
end 

struct LinearP{M, N} <: SympNet{M, N}
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

function initialparameters(backend::Backend, ::Type{T}, d::Activation{M, M}; rng::AbstractRNG = Random.default_rng(), init_scale = GlorotUniform()) where {M, T}
        a = KernelAbstractions.zeros(backend, T, M÷2, 1)
        init_scale(rng, a)
        return (scale = a,)
end

function initialparameters(backend::Backend, ::Type{T}, d::Linear{M, M}; rng::AbstractRNG = Random.default_rng(), init_weight = GlorotUniform()) where {M, T}
        K = KernelAbstractions.allocate(backend, T, d.second_dim÷2, M÷2)
        init_weight(rng, K)
        (weight=K, )
end

function parameterlength(d::Gradient{M, M}) where {M}
        d.second_dim÷2 * (M÷2 + 2)
end

function parameterlength(::Activation{M, M}) where {M}
        M÷2
end

function parameterlength(::Linear{M, M}) where {M}
        (M÷2)^2 
end

custom_mat_mul(weight::AbstractMatrix, x::AbstractVecOrMat) = weight*x 
function custom_mat_mul(weight::AbstractMatrix, x::AbstractArray{T, 3}) where T 
        mat_tensor_mul(weight, x)
end

custom_vec_mul(scale::AbstractVector, x::AbstractVecOrMat) = scale .* x 
function custom_vec_mul(scale::AbstractVector{T}, x::AbstractArray{T, 3}) where T 
        vec_tensor_mul(scale, x)
end


@inline function (d::ActivationQ{M, M})(x::AbstractVecOrMat, ps) where {M}
        size(x)[1] == M || error("Dimension mismatch.")
        N2 = M÷2
        q, p = assign_q_and_p(x, N2)
        return vcat(q + custom_vec_mul(ps.scale,d.activation.(p)), p)
end

@inline function (d::ActivationP{M, M})(x::AbstractVecOrMat, ps) where {M}
        size(x)[1] == M || error("Dimension mismatch.")
        N2 = M÷2 
        q, p = assign_q_and_p(x, N2)
        return vcat(q, p + custom_vec_mul(ps.scale,d.activation.(q)))
end

@inline function (d::GradientQ{M, M})(x::AbstractVecOrMat, ps) where {M}
        size(x)[1] == M || error("Dimension mismatch.")
        N2 = M÷2 
        q, p = assign_q_and_p(x, N2)
        return vcat(q + custom_mat_mul(ps.weight', (custom_vec_mul(ps.scale, d.activation.(custom_mat_mul(ps.weight, p) .+ ps.bias)))), p)
end

@inline function(d::GradientP{M, M})(x::AbstractVecOrMat, ps) where {M}
        size(x)[1] == M || error("Dimension mismatch.")
        N2 = M÷2 
        q, p = assign_q_and_p(x, N2)
        return vcat(q, p + custom_mat_mul(ps.weight', (custom_vec_mul(ps.scale, d.activation.(custom_mat_mul(ps.weight, q) .+ ps.bias)))))
end

@inline function (d::LinearQ{M, M})(x::AbstractVecOrMat, ps) where {M}
        size(x)[1] == M || error("Dimension mismatch.")
        N2 = M÷2 
        q, p = assign_q_and_p(x, N2)
        vcat(q + custom_mat_mul(ps.weight,p), p)
end

@inline function (d::LinearP{M, M})(x::AbstractVecOrMat, ps) where {M}
        size(x)[1] == M || error("Dimension mismatch.")
        N2 = M÷2 
        q, p = assign_q_and_p(x, N2)
        vcat(q, p + custom_mat_mul(ps.weight,q))
end