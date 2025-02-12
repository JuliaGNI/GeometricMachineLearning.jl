# """
# Defines a layer that performs simple multiplication with an element of the Stiefel manifold.
# """
struct StiefelLayer{M, N} <: ManifoldLayer{M, N} end

function StiefelLayer(n::Integer, N::Integer)
    StiefelLayer{n, N}()
end

function initialparameters(rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer, ::StiefelLayer{M,N}, backend::KernelAbstractions.Backend, ::Type{T}) where {M,N,T}
    weight = N > M ? KernelAbstractions.allocate(backend, T, N, M) : KernelAbstractions.allocate(backend, T, M, N)
    initializer(rng, weight)
    (weight = StiefelManifold(assign_columns(typeof(weight)(qr!(weight).Q), size(weight)...)),)
end

function parameterlength(::StiefelLayer{M, N}) where {M, N}
    N > M ? M * (M - 1) ÷ 2 + (N - M) * M : N * (N - 1) ÷ 2 + (M - N) * N
end