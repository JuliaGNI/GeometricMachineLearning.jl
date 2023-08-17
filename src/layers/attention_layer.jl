"""
MultiHeadAttention (MHA) serves as a preprocessing step in the transformer. It reweights the input vectors bases on correlations within those data. 
"""
struct Attention{M, N, Stiefel, Retraction, add_connection, FT} <: AbstractExplicitLayer{M, N}
    activation::FT
end

default_retr = Geodesic()
function orthonormal_activation_old(A::AbstractMatrix{T}) where T 
    exp(T(0.5)*(A - A'))
end
# TODO: This can be implemented more efficiently if you write one kernel for everything!
function orthonormal_activation_old(A::AbstractArray{T, 3}) where T 
    tensor_exponential(T(.5)*(A - tensor_transpose(A)))
end

function orthonormal_activation(A::AbstractMatrix{T}) where T 
    reshape(orthonormal_activation(reshape(A, size(A)..., 1)), size(A)...)
end

function orthonormal_activation(A::AbstractArray{T, 3}) where T 
    A_ut = upper_triangular_asymmetrize(A)
    fac = ceil(norm(A_ut)/size(A,3))
    expA = tensor_exponential(A_ut/fac)
    expA_mul = copy(expA)
    for _ in 2:fac 
        expA_mul = tensor_tensor_mul(expA, expA_mul)
    end
    expA_mul
end

function Attention(dim::Integer, activation=orthonormal_activation; Stiefel::Bool=false, retraction::AbstractRetraction=default_retr, add_connection::Bool=false)
    Attention{dim, dim, Stiefel, typeof(retraction), add_connection, typeof(activation)}(activation)
end

function parameterlength(::Attention{M, M, false}) where M
    2*M^2
end

function parameterlength(d::Attention{M, M, true}) where M
    M*(M-1)
end

function initialparameters(backend::KernelAbstractions.Backend, T::Type, d::Attention{M, M, false}; rng::AbstractRNG=Random.default_rng(), initializer::AbstractNeuralNetworks.AbstractInitializer=GlorotUniform()) where {M}
    # transformations for queries and keys.
    PQ_weight = KernelAbstractions.allocate(backend, T, M, M)
    PK_weight = KernelAbstractions.allocate(backend, T, M, M)
    initializer(rng, PQ_weight)
    initializer(rng, PK_weight)
    (PQ=PQ_weight, PK=PK_weight)
end

function initialparameters(backend::KernelAbstractions.Backend, T::Type, d::Attention{M, M, true}; rng::AbstractRNG=Random.default_rng(), initializer::AbstractNeuralNetworks.AbstractInitializer=GlorotUniform()) where {M}
    # projections for queries, keys and vectors.
    PQ_weight = rand(backend, rng, StiefelManifold{T}, M, M)
    PK_weight = rand(backend, rng, StiefelManifold{T}, M, M)
    (PQ=PQ_weight, PK=PK_weight)
end

function (d::Attention{M, M, Stiefel, Retraction, true})(x::AbstractMatrix{T}, ps::NamedTuple) where {M, Stiefel, Retraction, T}
    dim, input_length = size(x)
    @assert dim == M

    x + x*d.activation((ps.PQ'*x)'*(ps.PK'*x))
end

function (d::Attention{M, M, Stiefel, Retraction, false})(x::AbstractMatrix{T}, ps::NamedTuple) where {M, Stiefel, Retraction, T}
    dim, input_length = size(x)
    @assert dim == M

    x*d.activation((ps.PQ'*x)'*(ps.PK'*x))
end

function (d::Attention{M, M, Stiefel, Retraction, true})(x::AbstractArray{T, 3}, ps::NamedTuple) where {M, Stiefel, Retraction, T} 
    dim, input_length, number_data = size(x)
    @assert dim == M

    Q_tensor = mat_tensor_mul(ps.PQ', x)
    K_tensor = mat_tensor_mul(ps.PK', x)
    QK_tensor = tensor_transpose_tensor_mul(Q_tensor, K_tensor)
    x + tensor_tensor_mul(x, d.activation(QK_tensor))
end

function (d::Attention{M, M, Stiefel, Retraction, false})(x::AbstractArray{T, 3}, ps::NamedTuple) where {M, Stiefel, Retraction, T} 
    dim, input_length, number_data = size(x)
    @assert dim == M

    Q_tensor = mat_tensor_mul(ps.PQ', x)
    K_tensor = mat_tensor_mul(ps.PK', x)
    QK_tensor = tensor_transpose_tensor_mul(Q_tensor, K_tensor)
    tensor_tensor_mul(x, d.activation(QK_tensor))
end

@kernel function upper_triangular_asymmetrize_kernel!(output::AbstractArray{T, 3}, input::AbstractArray{T, 3}) where T 
    i,j,k = @index(Global, NTuple)
    if i < j
        output[i,j,k] = input[i,j,k]
    elseif i > j 
        output[i,j,k] = -input[j,i,k]
    end
end

function upper_triangular_asymmetrize(A::AbstractArray{T, 3}) where T 
    output = zero(A)
    backend = KernelAbstractions.get_backend(A)
    upper_triangular_asymmetrize! = upper_triangular_asymmetrize_kernel!(backend)
    upper_triangular_asymmetrize!(output, A, ndrange=size(A))
    output
end

@kernel function assign_upper_triangular_kernel!(output, input)
    i,j,k = @index(Global, NTuple)
    if i < j 
        output[i,j,k] += input[i,j,k]
    elseif i > j 
        output[j,i,k] -= input[i,j,k] 
    end
end

function assign_upper_triangular(A::AbstractArray{T, 3}) where T
    output = zero(A)
    backend = KernelAbstractions.get_backend(A)
    assign_upper_triangular! = assign_upper_triangular_kernel!(backend)
    assign_upper_triangular!(output, A, ndrange=size(A))
    output
end

function ChainRulesCore.rrule(::typeof(upper_triangular_asymmetrize), A::AbstractArray{T, 3}) where T 
    output = upper_triangular_asymmetrize(A)
    function upper_triangular_asymmetrize_pullback(output_diff)
        A_diff =  @thunk assign_upper_triangular(output_diff)
        return NoTangent(), A_diff
    end 
    return output, upper_triangular_asymmetrize_pullback 
end

#upper_triangular_asymmetrize(output_diff::Thunk) = Thunk(() -> upper_triangular_asymmetrize(unthunk(output_diff)))
assign_upper_triangular(output_diff::Thunk) = Thunk(() -> assign_upper_triangular(unthunk(output_diff)))