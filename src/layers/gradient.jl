@doc raw"""
The gradient layer from the SympNet paper (https://www.sciencedirect.com/science/article/abs/pii/S0893608020303063). 
Its components are of the form: 
$$
\begin{pmatrix}
        I & \nabla{}V \\ 0 & I 
\end{pmatrix},
$$
with $V(p) = \sum_ia_i\Sigma(\sum_jk_{ij}p_j+b_i)$, where $\Sigma$ is the antiderivative of the activation function $\sigma$. Such layers are by construction symplectic.
"""
abstract type Gradient{M, N, full_grad, TA} <: AbstractExplicitLayer{M, N} end

struct GradientQ{M, N, full_grad, TA} <: Gradient{M, N, full_grad, TA}
        second_dim::Integer
        activation::TA
end

struct GradientP{M, N, full_grad, TA} <: Gradient{M, N, full_grad, TA}
        second_dim::Integer
        activation::TA
end

# check: input is even; make dim2 an optional argument for full_grad=false
function Gradient(dim::Int, dim2::Int=dim, activation = identity; full_grad::Bool=true, change_q::Bool=true, allow_fast_activation::Bool=true)

        activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
        
        iseven(dim) && iseven(dim2) || error("Dimensions must be even!")
        dim2 ≥ dim || error("Second dimension should be bigger than the first!")

        if change_q
                return GradientQ{dim, dim, full_grad, typeof(activation)}(dim2, activation)
        else
                return GradientP{dim, dim, full_grad, typeof(activation)}(dim2, activation)
        end
end

GradientQ(dim::Int, dim2::Int=dim, activation = identity; full_grad::Bool=true, allow_fast_activation::Bool=true) = Gradient(dim, dim2, activation; full_grad = full_grad, change_q = true, allow_fast_activation = allow_fast_activation)
GradientP(dim::Int, dim2::Int=dim, activation = identity; full_grad::Bool=true, allow_fast_activation::Bool=true) = Gradient(dim, dim2, activation; full_grad = full_grad, change_q= false, allow_fast_activation = allow_fast_activation)


function initialparameters(backend::Backend, ::Type{T}, d::Gradient{M, N, true}; rng::AbstractRNG = Random.default_rng(), init_weight = GlorotUniform(), init_bias = ZeroInitializer(), init_scale = GlorotUniform()) where {M, N, T}
        K = KernelAbstractions.allocate(backend, T, d.second_dim÷2, M÷2)
        b = KernelAbstractions.allocate(backend, T, d.second_dim÷2)
        a = KernelAbstractions.allocate(backend, T, d.second_dim÷2)
        init_weight(rng, K)
        init_bias(rng, b)
        init_scale(rng, a)
        return (weight = K, bias = b, scale = a)
end

function initialparameters(backend::Backend, ::Type{T}, d::Gradient{M, N, false}; rng::AbstractRNG = Random.default_rng(), init_scale = GlorotUniform()) where {M, N, T}
        a = KernelAbstractions.zeros(backend, T, M÷2, 1)
        init_scale(rng, a)
        return (scale = a,)
end

function parameterlength(d::Gradient{M, M, full_grad}) where {M, full_grad}
        return full_grad ? d.second_dim÷2 * (M÷2 + 2) : M÷2
end


@inline function (d::GradientQ{M, M, false})(x::AbstractVecOrMat, ps) where {M}
        size(x)[1] == M || error("Dimension mismatch.")
        N2 = M÷2
        q, p = assign_q_and_p(x, N2)
        return vcat(q + ps.scale.*d.activation.(p), p)
end

@inline function (d::GradientP{M, M, false})(x::AbstractVecOrMat, ps) where {M}
        size(x)[1] == M || error("Dimension mismatch.")
        N2 = M÷2 
        q, p = assign_q_and_p(x, N2)
        return vcat(q, p + ps.scale.*d.activation.(q))
end

@inline function (d::GradientQ{M, M, true})(x::AbstractVecOrMat, ps) where {M}
        size(x)[1] == M || error("Dimension mismatch.")
        N2 = M÷2 
        q, p = assign_q_and_p(x, N2)
        return vcat(q + ps.weight' * (ps.scale .* d.activation.(ps.weight * p .+ ps.bias)), p)
end

@inline function(d::GradientP{M, M, true})(x::AbstractVecOrMat, ps) where {M}
        size(x)[1] == M || error("Dimension mismatch.")
        N2 = M÷2 
        q, p = assign_q_and_p(x, N2)
        return vcat(q, p + ps.weight' * (ps.scale .* d.activation.(ps.weight*q .+ ps.bias)))
end
