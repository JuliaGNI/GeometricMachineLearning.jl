abstract type VolumePreservingFeedForwardLayer{M, N} <: AbstractExplicitLayer{M, N} end 

struct VolumePreservingLowerLayer{M, N, AT} <: VolumePreservingFeedForwardLayer{M, N}
    activation::AT

    function VolumePreservingLowerLayer(sys_dim::Int, activation=tanh)
        new{sys_dim, sys_dim, typeof(activation)}(activation)
    end
end 

struct VolumePreservingUpperLayer{M, N, AT} <: VolumePreservingFeedForwardLayer{M, N}
    activation::AT

    function VolumePreservingUpperLayer(sys_dim::Int, activation=tanh)
        new{sys_dim, sys_dim, typeof(activation)}(activation)
    end
end 

parameterlength(::VolumePreservingFeedForwardLayer{M, M}) where M = M * (M - 1) ÷ 2 

function initialparameters(backend::Backend, ::Type{T}, d::VolumePreservingLowerLayer{M, M}; rng::AbstractRNG = Random.default_rng(), init_weight! = GlorotUniform(), init_bias! = init_weight!) where {M, T}
    S = KernelAbstractions.allocate(backend, T, parameterlength(d))
    b = KernelAbstractions.allocate(backend, T, M)
    init_weight!(rng, S)
    init_bias!(rng, b)

    (weight = LowerTriangular(S, M), bias = b)
end 

function initialparameters(backend::Backend, ::Type{T}, d::VolumePreservingUpperLayer{M, M}; rng::AbstractRNG = Random.default_rng(), init_weight! = GlorotUniform(), init_bias! = init_weight!) where {M, T}
    S = KernelAbstractions.allocate(backend, T, parameterlength(d))
    b = KernelAbstractions.allocate(backend, T, M)
    init_weight!(rng, S)
    init_bias!(rng, b)
    
    (weight = UpperTriangular(S, M), bias = b)
end 

function (d::VolumePreservingFeedForwardLayer)(x::AbstractArray{T, 3}, ps) where T
    x + d.activation.(mat_tensor_mul(ps.weight, x) .+ ps.bias)
end

function (d::VolumePreservingFeedForwardLayer)(x::AbstractVecOrMat{T}, ps) where T 
    x + d.activation.(ps.weight * x .+ ps.bias)
end