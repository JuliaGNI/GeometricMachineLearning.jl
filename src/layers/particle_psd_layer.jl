# The supertype of `ParticlePSDLayer` and `TransposeParticlePSDLayer`, which share every method but
# the choice of `Φ` or `Φ'`.
abstract type _ParticlePSD{M, N} <: AbstractExplicitLayer{M, N} end

@doc raw"""
    ParticlePSDLayer(M, d_in, d_out)

Make an instance of `ParticlePSDLayer`, the cotangent lift of ``I_M \otimes \Phi`` for `M` particles, which raises the per-particle dimension from `d_in` to `d_out` (`1 <= d_in <= d_out`).

The layer maps the state of `M` particles in ``\mathbb{R}^{d_\mathrm{in}}`` (phase-space dimension ``2Md_\mathrm{in}``) to `M` particles in ``\mathbb{R}^{d_\mathrm{out}}`` by applying one ``\Phi \in \mathrm{St}(d_\mathrm{in}, d_\mathrm{out})``, shared by all particles, to every position and every momentum:

```math
(q_i, p_i) \mapsto (\Phi q_i, \Phi p_i), \qquad i = 1, \ldots, M,
```

i.e. it multiplies by

```math
A = \begin{bmatrix} I_M \otimes \Phi & \mathbb{O} \\ \mathbb{O} & I_M \otimes \Phi \end{bmatrix}.
```

Because ``\Phi^T\Phi = I``, ``A`` satisfies ``A^T\mathbb{J}_{2Md_\mathrm{out}}A = \mathbb{J}_{2Md_\mathrm{in}}``, and because every particle is lifted with the same ``\Phi``, the layer commutes with every relabelling of the particles. It is the per-particle counterpart of [`PSDLayer`](@ref); its left inverse is [`TransposeParticlePSDLayer`](@ref), which shares the parameters `(weight = Φ,)`.

The input is a vector or an array whose first axis has length ``2Md_\mathrm{in}``, or a `NamedTuple` `(q, p)` whose halves have first axis ``Md_\mathrm{in}``, in the flattened particle layout of [`PairwiseSymplecticLayer`](@ref): particle ``i`` occupies the indices ``(i - 1)d_\mathrm{in} + 1, \ldots, id_\mathrm{in}`` of `q` and of `p`. A `(q, p)` pair of particle arrays of size `(d_in, M)` or `(d_in, M, B)` is also accepted and returns arrays of size `(d_out, M)` or `(d_out, M, B)`.
"""
struct ParticlePSDLayer{M, N} <: _ParticlePSD{M, N}
    n_particles::Int
end

@doc raw"""
    TransposeParticlePSDLayer(M, d_in, d_out)

Make an instance of `TransposeParticlePSDLayer`, which lowers the per-particle dimension of `M` particles from `d_in` to `d_out` (`1 <= d_out <= d_in`) by applying ``\Phi^T`` to every position and every momentum:

```math
(q_i, p_i) \mapsto (\Phi^T q_i, \Phi^T p_i), \qquad i = 1, \ldots, M,
```

with ``\Phi \in \mathrm{St}(d_\mathrm{out}, d_\mathrm{in})``. It is the analytic transpose, and the left inverse on its image, of [`ParticlePSDLayer`](@ref)`(M, d_out, d_in)`, with which it shares the parameters `(weight = Φ,)`; [`inverse_layer`](@ref) maps one to the other. Inputs are as for [`ParticlePSDLayer`](@ref).
"""
struct TransposeParticlePSDLayer{M, N} <: _ParticlePSD{M, N}
    n_particles::Int
end

function _check_particle_psd_args(name::String, M::Integer, d_in::Integer, d_out::Integer)
    (M >= 1 && d_in >= 1 && d_out >= 1) ||
        throw(ArgumentError(
            "$name requires M >= 1, d_in >= 1 and d_out >= 1, got M = $M, d_in = $d_in, d_out = $d_out."))
    nothing
end

function ParticlePSDLayer(M::Integer, d_in::Integer, d_out::Integer)
    _check_particle_psd_args("ParticlePSDLayer", M, d_in, d_out)
    d_in <= d_out ||
        throw(ArgumentError(
            "ParticlePSDLayer requires d_in <= d_out, got d_in = $d_in, d_out = $d_out."))
    Mp = Int(M)
    ParticlePSDLayer{2 * Mp * Int(d_in), 2 * Mp * Int(d_out)}(Mp)
end

function TransposeParticlePSDLayer(M::Integer, d_in::Integer, d_out::Integer)
    _check_particle_psd_args("TransposeParticlePSDLayer", M, d_in, d_out)
    d_in >= d_out ||
        throw(ArgumentError(
            "TransposeParticlePSDLayer requires d_in >= d_out, got d_in = $d_in, d_out = $d_out."))
    Mp = Int(M)
    TransposeParticlePSDLayer{2 * Mp * Int(d_in), 2 * Mp * Int(d_out)}(Mp)
end

# The per-particle input and output dimensions.
function _particle_dims(l::_ParticlePSD{M, N}) where {M, N}
    (M ÷ (2 * l.n_particles), N ÷ (2 * l.n_particles))
end

# The weight Φ ∈ St(small, large) is stored as a `large × small` matrix.
function _particle_stiefel_size(l::_ParticlePSD)
    d_in, d_out = _particle_dims(l)
    (max(d_in, d_out), min(d_in, d_out))
end

function parameterlength(l::_ParticlePSD)
    N, n = _particle_stiefel_size(l)
    N * n - (n * (n + 1)) ÷ 2
end

function initialparameters(
        rng::AbstractRNG, initializer::AbstractNeuralNetworks.Initializer,
        l::_ParticlePSD, backend::KernelAbstractions.Backend, ::Type{T}) where {T}
    N, n = _particle_stiefel_size(l)
    weight = KernelAbstractions.allocate(backend, T, N, n)
    (weight = StiefelManifold(orthonormal_columns() do
        initializer(rng, weight)
        weight
    end),)
end

_particle_weight(::ParticlePSDLayer, ps::NamedTuple) = ps.weight
_particle_weight(::TransposeParticlePSDLayer, ps::NamedTuple) = ps.weight'

# Apply `W` to every particle of `x`. A first axis of length `M * d_in` is the flattened layout,
# in which particle `i` is the block `(i - 1) * d_in + 1:i * d_in`; otherwise `x` is a particle
# array of size `(d_in, M, ...)`. For `M = 1` the two readings give the same result.
function _apply_particle_weight(
        l::_ParticlePSD, W::AbstractMatrix, x::AbstractArray)
    Mp = l.n_particles
    d_in, d_out = _particle_dims(l)
    if size(x, 1) == Mp * d_in
        reshape(_mul(W, reshape(x, d_in, :)), Mp * d_out, Base.tail(size(x))...)
    elseif ndims(x) >= 2 && size(x, 1) == d_in && size(x, 2) == Mp
        _mul(W, x)
    else
        throw(DimensionMismatch(
            "$(nameof(typeof(l))) expects q and p with first dimension $(Mp * d_in) or of size ($d_in, $Mp, ...), got $(size(x))."))
    end
end

function (l::_ParticlePSD)(qp::NamedTuple{(:q, :p), Tuple{AT1, AT2}},
        ps::NamedTuple) where {AT1 <: AbstractArray, AT2 <: AbstractArray}
    size(qp.q) == size(qp.p) ||
        throw(DimensionMismatch(
            "$(nameof(typeof(l))) expects q and p of matching size, got $(size(qp.q)) and $(size(qp.p))."))
    W = _particle_weight(l, ps)
    (q = _apply_particle_weight(l, W, qp.q), p = _apply_particle_weight(l, W, qp.p))
end

function (l::_ParticlePSD{M})(x::AbstractArray, ps::NamedTuple) where {M}
    size(x, 1) == M ||
        throw(DimensionMismatch(
            "$(nameof(typeof(l))) expects first dimension $M, got $(size(x, 1))."))
    qp = assign_q_and_p(x, M ÷ 2)
    out = l(qp, ps)
    vcat(out.q, out.p)
end

inverse_layer(l::ParticlePSDLayer{M, N}) where {M, N} = TransposeParticlePSDLayer{N, M}(l.n_particles)
inverse_layer(l::TransposeParticlePSDLayer{M, N}) where {M, N} = ParticlePSDLayer{N, M}(l.n_particles)
inverse_layer(::Type{ParticlePSDLayer{M, N}}) where {M, N} = TransposeParticlePSDLayer{N, M}
inverse_layer(::Type{TransposeParticlePSDLayer{M, N}}) where {M, N} = ParticlePSDLayer{N, M}
inverse_layer(::Type{ParticlePSDLayer}) = TransposeParticlePSDLayer
inverse_layer(::Type{TransposeParticlePSDLayer}) = ParticlePSDLayer
