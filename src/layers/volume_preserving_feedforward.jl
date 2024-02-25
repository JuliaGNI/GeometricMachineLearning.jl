abstract type VolumePreservingFeedForwardLayer{M, N, bias} <: AbstractExplicitLayer{M, N} end 

struct VolumePreservingLowerLayer{M, N, bias, AT} <: VolumePreservingFeedForwardLayer{M, N, bias}
    activation::AT

    function VolumePreservingLowerLayer(sys_dim::Int, activation=tanh; include_bias=true)
        if include_bias
            return new{sys_dim, sys_dim, :bias, typeof(activation)}(activation)
        else 
            return new{sys_dim, sys_dim, :no_bias, typeof(activation)}(activation)
        end
    end
end 

struct VolumePreservingUpperLayer{M, N, bias, AT} <: VolumePreservingFeedForwardLayer{M, N, bias}
    activation::AT

    function VolumePreservingUpperLayer(sys_dim::Int, activation=tanh; include_bias=true)
        if include_bias
            return new{sys_dim, sys_dim, :bias, typeof(activation)}(activation)
        else 
            return new{sys_dim, sys_dim, :no_bias, typeof(activation)}(activation)
        end
    end
end 

parameterlength(::VolumePreservingFeedForwardLayer{M, M, :no_bias}) where M = M * (M - 1) ÷ 2 
parameterlength(::VolumePreservingFeedForwardLayer{M, M, :bias}) where M = M * (M - 1) ÷ 2 + M

function initialparameters(backend::Backend, ::Type{T}, d::VolumePreservingLowerLayer{M, M, :bias}; rng::AbstractRNG = Random.default_rng(), init_weight! = GlorotUniform(), init_bias! = ZeroInitializer()) where {M, T}
    S = KernelAbstractions.allocate(backend, T, parameterlength(d))
    b = KernelAbstractions.allocate(backend, T, M)
    init_weight!(rng, S)
    init_bias!(rng, b)

    (weight = LowerTriangular(S, M), bias = b)
end 

function initialparameters(backend::Backend, ::Type{T}, d::VolumePreservingUpperLayer{M, M, :bias}; rng::AbstractRNG = Random.default_rng(), init_weight! = GlorotUniform(), init_bias! = ZeroInitializer()) where {M, T}
    S = KernelAbstractions.allocate(backend, T, parameterlength(d))
    b = KernelAbstractions.allocate(backend, T, M)
    init_weight!(rng, S)
    init_bias!(rng, b)
    
    (weight = UpperTriangular(S, M), bias = b)
end 

function initialparameters(backend::Backend, ::Type{T}, d::VolumePreservingLowerLayer{M, M, :no_bias}; rng::AbstractRNG = Random.default_rng(), init_weight! = GlorotUniform(), init_bias! = ZeroInitializer()) where {M, T}
    S = KernelAbstractions.allocate(backend, T, parameterlength(d))
    init_weight!(rng, S)

    (weight = LowerTriangular(S, M), )
end 

function initialparameters(backend::Backend, ::Type{T}, d::VolumePreservingUpperLayer{M, M, :no_bias}; rng::AbstractRNG = Random.default_rng(), init_weight! = GlorotUniform(), init_bias! = ZeroInitializer()) where {M, T}
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

function (d::VolumePreservingFeedForwardLayer{M, M, :bias})(x::AbstractVecOrMat{T}, ps) where {T, M} 
    x + d.activation.(ps.weight * x .+ ps.bias)
end

function (d::VolumePreservingFeedForwardLayer{M, M, :no_bias})(x::AbstractVecOrMat{T}, ps) where {T, M} 
    x + d.activation.(ps.weight * x)
end