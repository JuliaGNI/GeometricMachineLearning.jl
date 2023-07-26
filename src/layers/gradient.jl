#=
If full_grad is true, then the Gradient layer also has a bias
=#

#activation layer
struct Gradient{M, N, full_grad, change_q, TA} <: AbstractExplicitLayer{M, N}
        activation::TA
end

#check: input is even; make dim2 an optional argument for full_grad=false
function Gradient(dim::Int, dim2::Int=dim, activation = identity; full_grad::Bool=true, change_q::Bool=true, allow_fast_activation::Bool=true)

        activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
        
        iseven(dim) && iseven(dim2) || error("Dimensions must be even!")
        dim2 ≥ dim || error("Second dimension should be bigger than the first!")

        return Gradient{dim, dim2, full_grad, change_q, typeof(activation)}(activation)
end


struct GlorotUniform <: AbstractNeuralNetworks.AbstractInitializer end

(::GlorotUniform)(rng, x) = x .= sqrt(24.0 / sum(size(x))) .* (rand(rng, eltype(x), size(x)...) .- 0.5) 



function initialparameters(backend::Backend, ::Type{T}, d::Gradient{M, N, true}; rng::AbstractRNG = Random.default_rng(), init_weight = GlorotUniform(), init_bias = ZeroInitializer(), init_scale = GlorotUniform()) where {M, N, T}
        W = KernelAbstractions.zeros(backend, T, N÷2, M÷2)
        b = KernelAbstractions.zeros(backend, T, N÷2, 1)
        K = KernelAbstractions.zeros(backend, T, N÷2, 1)
        init_weight(rng, W)
        init_bias(rng, b)
        init_scale(rng, K)
        return (weight = W, bias = b, scale = K)
end

function initialparameters(backend::Backend, ::Type{T}, d::Gradient{M, N, false}; rng::AbstractRNG = Random.default_rng(), init_scale = GlorotUniform()) where {M, N, T}
        K = KernelAbstractions.zeros(backend, T, M÷2, 1)
        init_scale(rng, K)
        return (scale = K,)
end

function parameterlength(::Gradient{M, N, full_grad}) where {M, N, full_grad}
        return full_grad ? N÷2 * (M÷2 + 2) : M÷2
end


@inline function (d::Gradient{M, N, false,true})(x::AbstractVecOrMat, ps::NamedTuple)  where {M, N}
        size(x)[1] == M || error("Dimension mismatch.")
        return vcat(x[1:(M÷2)] + ps.scale.*d.activation.(x[(M÷2+1):M]), x[(M÷2+1):M])
end

@inline function (d::Gradient{M, N, false,false})(x::AbstractVecOrMat, ps::NamedTuple) where {M, N}
        size(x)[1] == M || error("Dimension mismatch.")
        return vcat(x[1:(M÷2)], x[(M÷2+1):M] + ps.scale.* d.activation.(x[1:(M÷2)]))
end

@inline function (d::Gradient{M, N, true,true})(x::AbstractVecOrMat, ps::NamedTuple) where {M, N}
        size(x)[1] == M || error("Dimension mismatch.")
        return vcat(x[1:(M÷2)] + ps.weight' * (ps.scale .* d.activation.(ps.weight * x[(M÷2+1):M] .+ vec(ps.bias))), x[(M÷2+1):M])
end

@inline function(d::Gradient{M, N, true,false})(x::AbstractVecOrMat, ps::NamedTuple) where {M, N}
        size(x)[1] == M || error("Dimension mismatch.")
        return vcat(x[1:(M÷2)], x[(M÷2+1):M] + ps.weight' * (ps.scale .* d.activation(ps.weight*x[1:(M÷2)] .+ vec(ps.bias))))
end

#######
#this is for GPU support (doesn't support indexing arrays); for now only CUDA!!

function assign_first_half!(q, x)
        i = CUDA.threadIdx().x
        q[i] = x[i]
        return 
end

function assign_second_half!(p, x, N)
        i = CUDA.threadIdx().x
        p[i] = x[i+N]
        return 
end

function assign_q_and_p(x, N)
        q = CUDA.zeros(eltype(x), N)
        p = CUDA.zeros(eltype(x), N)
        CUDA.@cuda threads=N assign_first_half!(q, x)
        CUDA.@cuda threads=N assign_second_half!(p, x, N)
        q, p
end

@inline function (d::Gradient{M, N, false,true})(x::AbstractGPUVecOrMat, ps) where {M, N}
        size(x)[1] == M || error("Dimension mismatch.")
        N2 = M÷2
        q, p = assign_q_and_p(x, N2)
        return vcat(q + ps.scale.*d.activation.(p), p)
end

@inline function (d::Gradient{M, N, false,false})(x::AbstractGPUVecOrMat, ps) where {M, N}
        size(x)[1] == M || error("Dimension mismatch.")
        N2 = M÷2 
        q, p = assign_q_and_p(x, N2)
        return vcat(q, p + ps.scale.*d.activation.(q))
end

@inline function (d::Gradient{M, N, true,true})(x::AbstractGPUVecOrMat, ps) where {M, N}
        size(x)[1] == M || error("Dimension mismatch.")
        N2 = M÷2 
        q, p = assign_q_and_p(x, N2)
        return vcat(q + ps.weight' * (ps.scale .* d.activation.(ps.weight * p .+ vec(ps.bias))), p)
end

@inline function(d::Gradient{M, N, true,false})(x::AbstractGPUVecOrMat, ps) where {M, N}
        size(x)[1] == M || error("Dimension mismatch.")
        N2 = M÷2 
        q, p = assign_q_and_p(x, N2)
        return vcat(q, p + ps.weight' * (ps.scale .* d.activation(ps.weight*q .+ vec(ps.bias))))
end

@inline function (d::Gradient{M, N, false,true})(x::AbstractVecOrMat, ps::Tuple) where {M, N}
        size(x)[1] == M || error("Dimension mismatch.")
        return vcat(x[1:(M÷2)] + ps[3].*d.activation.(x[(M÷2+1):M]), x[(M÷2+1):M])
end

@inline function (d::Gradient{M, N, false,false})(x::AbstractVecOrMat, ps::Tuple) where {M, N}
        size(x)[1] == M || error("Dimension mismatch.")
        return vcat(x[1:(M÷2)], x[(M÷2+1):M] + ps[3].* d.activation.(x[1:(M÷2)]))
end

@inline function (d::Gradient{M, N, true,true})(x::AbstractVecOrMat, ps::Tuple) where {M, N}
        size(x)[1] == M || error("Dimension mismatch.")
        return vcat(x[1:(M÷2)] + ps[1]' * (ps[3] .* d.activation.(ps[1] * x[(M÷2+1):M] .+ vec(ps[2]))), x[(M÷2+1):M])
end

@inline function(d::Gradient{M, N, true,false})(x::AbstractVecOrMat, ps::Tuple) where {M, N}
        size(x)[1] == M || error("Dimension mismatch.")
        return vcat(x[1:(M÷2)], x[(M÷2+1):M] + ps[1]' * (ps[3] .* d.activation(ps[1]*x[1:(M÷2)] .+ vec(ps[2]))))
end


