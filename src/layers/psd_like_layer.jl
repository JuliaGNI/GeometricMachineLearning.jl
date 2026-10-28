@doc raw"""
    PSDLayer(input_dim, output_dim)

Make an instance of `PSDLayer`.

This is a PSD-like layer used for symplectic autoencoders. 
One layer has the following shape:

```math
A = \begin{bmatrix} \Phi & \mathbb{O} \\ \mathbb{O} & \Phi \end{bmatrix},
```
where ``\Phi`` is an element of the Stiefel manifold ``St(n, N)``.
"""
struct PSDLayer{M, N} <: AbstractExplicitLayer{M, N} end

function PSDLayer(M::Integer, N::Integer)
    @assert iseven(M)
    @assert iseven(N)
    PSDLayer{M, N}()
end

function parameterlength(::PSDLayer{M, N}) where {M, N}
    M2 = M ÷ 2
    N2 = N ÷ 2
    N > M ? M2 * N2 - (M2 * (M2 + 1)) ÷ 2 : N2 * M2 - (N2 * (N2 + 1)) ÷ 2
end

function initialparameters(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        ::PSDLayer{M, N}, backend::KernelAbstractions.Backend, ::Type{T}) where {M, N, T}
    weight = N > M ? KernelAbstractions.allocate(backend, T, N ÷ 2, M ÷ 2) :
             KernelAbstractions.allocate(backend, T, M ÷ 2, N ÷ 2)
    (weight = StiefelManifold(orthonormal_columns() do
        initializer(rng, weight)
        weight
    end),)
end

function (::PSDLayer{M, N})(qp::NamedTuple{(:q, :p), Tuple{AT1, AT2}},
        ps::NamedTuple) where {M, N, AT1 <: AbstractArray, AT2 <: AbstractArray}
    N > M ? (q = _mul(ps.weight, qp.q), p = _mul(ps.weight, qp.p)) :
    (q = _mul(ps.weight', qp.q), p = _mul(ps.weight', qp.p))
end

function (l::PSDLayer{M, N})(x::AbstractArray, ps::NamedTuple) where {M, N}
    dim = size(x, 1)
    @assert M == dim

    qp = assign_q_and_p(x, dim÷2)
    _vcat(l(qp, ps))
end

@doc raw"""
    TransposePSDLayer(input_dim, output_dim)

Make an instance of `TransposePSDLayer`.

This layer is the analytic transpose (and left inverse on the image of [`PSDLayer`](@ref)) of `PSDLayer(output_dim, input_dim)`, sharing the same parameter tuple `ps = (weight = Φ,)` with ``\Phi \in \mathrm{St}(n, N)``:

```math
A^T = \begin{bmatrix} \Phi^T & \mathbb{O} \\ \mathbb{O} & \Phi^T \end{bmatrix}.
```
"""
struct TransposePSDLayer{M, N} <: AbstractExplicitLayer{M, N}
    function TransposePSDLayer{M, N}() where {M, N}
        M isa Integer && N isa Integer ||
            throw(ArgumentError("TransposePSDLayer dimensions ($M, $N) must be integers."))
        (M > 0 && N > 0 && iseven(M) && iseven(N)) ||
            throw(ArgumentError("TransposePSDLayer dimensions ($M, $N) must be positive even integers."))
        new{Int(M), Int(N)}()
    end
end

function TransposePSDLayer(M::Integer, N::Integer)
    TransposePSDLayer{Int(M), Int(N)}()
end

function parameterlength(::TransposePSDLayer{M, N}) where {M, N}
    parameterlength(PSDLayer{N, M}())
end

function initialparameters(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        ::TransposePSDLayer{M, N}, backend::KernelAbstractions.Backend,
        ::Type{T}) where {M, N, T}
    initialparameters(rng, initializer, PSDLayer{N, M}(), backend, T)
end

function (::TransposePSDLayer{M, N})(qp::NamedTuple{(:q, :p), Tuple{AT1, AT2}},
        ps::NamedTuple) where {M, N, AT1 <: AbstractArray, AT2 <: AbstractArray}
    size(qp.q, 1) == size(qp.p, 1) == M ÷ 2 || error("Dimension mismatch.")
    M > N ? (q = _mul(ps.weight', qp.q), p = _mul(ps.weight', qp.p)) :
    (q = _mul(ps.weight, qp.q), p = _mul(ps.weight, qp.p))
end

function (l::TransposePSDLayer{M, N})(x::AbstractArray, ps::NamedTuple) where {M, N}
    dim = size(x, 1)
    @assert M == dim

    qp = assign_q_and_p(x, dim ÷ 2)
    _vcat(l(qp, ps))
end
