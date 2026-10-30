struct ReducedVFCache{T, JR}
    ξ_buf::Vector{T}
    v_intermediate::Vector{T}
    f_intermediate::Vector{T}
    z_bar_buf::Vector{T}
    jac_result::JR
    dual_cache::IdDict{DataType, Any}
end

function ReducedVFCache(::Type{T}, N2::Int, n2::Int) where {T}
    ξ_buf = zeros(T, 2 * n2)
    v_int = zeros(T, N2)
    f_int = zeros(T, N2)
    z_bar = zeros(T, 2 * N2)
    jac_res = DiffResults.JacobianResult(zeros(T, 2 * N2), ξ_buf)
    ReducedVFCache{T, typeof(jac_res)}(
        ξ_buf, v_int, f_int, z_bar, jac_res, IdDict{DataType, Any}())
end

function _get_vf_cache(
        cache::ReducedVFCache{T0}, ::Type{T}, N2::Int, n2::Int) where {T0, T}
    if T === T0
        return cache
    end
    T <: AbstractFloat && throw(
        ArgumentError(
        "The reduced vector field got a state of element type $T, but its cache has element type $T0."
    ),
    )
    get!(cache.dual_cache, T) do
        ReducedVFCache(T, N2, n2)
    end::ReducedVFCache{T}
end

function _check_pullback_mode(pullback_mode::Symbol)
    (pullback_mode === :jacobian || pullback_mode === :vjp) ||
        throw(
            ArgumentError(
            "pullback_mode must be :jacobian or :vjp, got :$pullback_mode."),
        )
    nothing
end

function _add_ref!(
        z::AbstractVector, q_ref::AbstractVector, p_ref::AbstractVector, N2::Int)
    @inbounds for i in 1:N2
        z[i] += q_ref[i]
        z[N2 + i] += p_ref[i]
    end
    z
end

function _add_ref!(z::AbstractVector, q_ref::AbstractVector, ::Nothing, N2::Int)
    @inbounds for i in 1:N2
        z[i] += q_ref[i]
    end
    z
end

function _add_ref!(z::AbstractVector, ::Nothing, p_ref::AbstractVector, N2::Int)
    @inbounds for i in 1:N2
        z[N2 + i] += p_ref[i]
    end
    z
end

_add_ref!(z::AbstractVector, ::Nothing, ::Nothing, ::Int) = z

@doc raw"""
    HRedSys

`HRedSys` computes the reconstructed dynamics in the full system based on the reduced one. Optionally it can be compared to the FOM solution.

It can be called using two different constructors.

# Constructors

The standard constructor is:

```julia
HRedSys(N, n, encoder, decoder, v_full, f_full, h_full, timespan, timestep, ics; integrator, parameters, q_ref, p_ref, pullback_mode, v_reduced, f_reduced, h_reduced)
``` 
where 
- `encoder`: a function ``\mathbb{R}^{2N}\mapsto{}\mathbb{R}^{2n}``
- `decoder`: a (differentiable) function ``\mathbb{R}^{2n}\mapsto\mathbb{R}^{2N}``
- `v_full`: a (differentiable) mapping defined the same way as in GeometricIntegrators.
- `f_full`: a (differentiable) mapping defined the same way as in GeometricIntegrators.
- `v_reduced`: a (differentiable) mapping defined the same way as in GeometricIntegrators.
- `f_reduced`: a (differentiable) mapping defined the same way as in GeometricIntegrators.
- `integrator`: is used for integrating the reduced system. It is a required keyword argument, here and in the two `HRedSys(odeensemble, encoder, decoder)` constructors.
- `parameters`: the parameters (a NamedTuple or `NullParameters()`) that parametrize the vector fields (the same for full_vector_field and reduced_vector_field). It defaults to `NullParameters()`; the two constructors that take a `HODEProblem` or a `HODEEnsemble` take it from the problem.
- `q_ref`, `p_ref`: optional reference configuration and momentum vectors in ``\mathbb{R}^N`` (defaulting to `zero(ics.q)` and `zero(ics.p)`). The decoder lifts ``\xi = (\tilde{q}, \tilde{p})`` as ``(q, p) = (q_{\mathrm{ref}}, p_{\mathrm{ref}}) + \Psi^{\mathrm{dec}}(\tilde{q}, \tilde{p})`` and the encoder encodes ``(q - q_{\mathrm{ref}}, p - p_{\mathrm{ref}})``.
- `pullback_mode`: `:jacobian` (default) or `:vjp`, forwarded to [`build_v_reduced`](@ref) and [`build_f_reduced`](@ref).
- `timespan`: a tuple `(t₀, tₗ)` that specifies start and end point of the time interval over which integration is performed. 
- `timestep`: the time step 
- `ics`: the initial condition for the big system.

The other, and more convenient, constructor is:

```julia
HRedSys(odeensemble, encoder, decoder; integrator, q_ref, p_ref, pullback_mode)
```
where `odeensemble` can be a `HODEEnsemble` or a `HODEProblem`. `encoder` and `decoder` have to be neural networks.
The keyword argument `integrator` is required here as in the standard constructor: no constructor picks one for the caller.
Internally this calls the functions [`build_v_reduced`](@ref) and [`build_f_reduced`](@ref) in order to build the reduced vector fields.
"""
struct HRedSys{
    T <: Number,
    ET <: NeuralNetwork{<:SymplecticEncoder},
    DT <: NeuralNetwork{<:SymplecticDecoder},
    VFT <: Function,
    FFT <: Function,
    HFT <: Function,
    VRT <: Function,
    FRT <: Function,
    HRT <: Function,
    InT, # type of integrator
    PT,  # type of parameters (of the vector field/hamiltonian)
    IT <: NamedTuple{(:q, :p), Tuple{ST, ST}} where ST <: StateVariable{T},
    TT <: Tuple, # type of timespan
    RT <: AbstractVector{T},
    JR
}
    N::Int
    n::Int
    encoder::ET
    decoder::DT
    v_full::VFT
    f_full::FFT
    h_full::HFT
    v_reduced::VRT
    f_reduced::FRT
    h_reduced::HRT
    integrator::InT
    parameters::PT
    timespan::TT
    timestep::T
    ics::IT
    q_ref::RT
    p_ref::RT
    pullback_mode::Symbol
    ξ_buf::Vector{T}
    v_intermediate::Vector{T}
    f_intermediate::Vector{T}
    jac_result::JR
    dual_cache::IdDict{DataType, Any}
end

function HRedSys(N::Integer, n::Integer, encoder::NeuralNetwork{<:SymplecticEncoder},
        decoder::NeuralNetwork{<:SymplecticDecoder}, v_full, f_full,
        h_full, timespan::Tuple, timestep::T, ics;
        parameters = NullParameters(),
        q_ref::AbstractVector = zeros(eltype(ics.q), length(ics.q)),
        p_ref::AbstractVector = zeros(eltype(ics.p), length(ics.p)),
        pullback_mode::Symbol = :jacobian,
        cache::ReducedVFCache = ReducedVFCache(
            eltype(ics.q), decoder.architecture.full_dim ÷ 2,
            decoder.architecture.reduced_dim ÷ 2),
        v_reduced = nothing,
        f_reduced = nothing,
        h_reduced = nothing,
        integrator) where {T <: Real}
    _check_pullback_mode(pullback_mode)
    N_ics = length(ics.q)
    (length(q_ref) == N_ics && length(p_ref) == N_ics) ||
        throw(
            DimensionMismatch(
            "q_ref and p_ref must have length N = $N_ics, got $(length(q_ref)) and $(length(p_ref))."
        ),
        )
    T_state = eltype(ics.q)
    q_ref_vec = Vector{T_state}(q_ref)
    p_ref_vec = Vector{T_state}(p_ref)
    v_red = isnothing(v_reduced) ?
            build_v_reduced(
        v_full, f_full, decoder;
        pullback_mode = pullback_mode, q_ref = q_ref_vec, p_ref = p_ref_vec,
        cache = cache) :
            v_reduced
    f_red = isnothing(f_reduced) ?
            build_f_reduced(
        v_full, f_full, decoder;
        pullback_mode = pullback_mode, q_ref = q_ref_vec, p_ref = p_ref_vec,
        cache = cache) :
            f_reduced
    h_red = isnothing(h_reduced) ?
            build_h_reduced(h_full, decoder; q_ref = q_ref_vec, p_ref = p_ref_vec) :
            h_reduced
    HRedSys{typeof(timestep), typeof(encoder), typeof(decoder), typeof(v_full),
        typeof(f_full), typeof(h_full), typeof(v_red), typeof(f_red),
        typeof(h_red), typeof(integrator), typeof(parameters), typeof(ics),
        typeof(timespan), typeof(q_ref_vec), typeof(cache.jac_result)}(
        N, n, encoder, decoder, v_full, f_full, h_full, v_red, f_red,
        h_red, integrator, parameters, timespan, timestep, ics,
        q_ref_vec, p_ref_vec, pullback_mode,
        cache.ξ_buf, cache.v_intermediate, cache.f_intermediate,
        cache.jac_result, cache.dual_cache)
end

function HRedSys(odeproblem::HODEProblem, encoder::NeuralNetwork{<:SymplecticEncoder},
        decoder::NeuralNetwork{<:SymplecticDecoder};
        integrator,
        q_ref::AbstractVector = zeros(eltype(odeproblem.ics.q), length(odeproblem.ics.q)),
        p_ref::AbstractVector = zeros(eltype(odeproblem.ics.p), length(odeproblem.ics.p)),
        pullback_mode::Symbol = :jacobian)
    N = encoder.architecture.full_dim
    n = encoder.architecture.reduced_dim
    v_eq = odeproblem.equation.v
    f_eq = odeproblem.equation.f
    h_eq = odeproblem.equation.hamiltonian
    HRedSys(
        N, n, encoder, decoder, v_eq, f_eq, h_eq, odeproblem.timespan, odeproblem.timestep,
        odeproblem.ics;
        parameters = odeproblem.parameters,
        q_ref = q_ref,
        p_ref = p_ref,
        pullback_mode = pullback_mode,
        integrator = integrator)
end

function HRedSys(odeensemble::HODEEnsemble, encoder::NeuralNetwork{<:SymplecticEncoder},
        decoder::NeuralNetwork{<:SymplecticDecoder};
        integrator,
        q_ref::AbstractVector = zeros(
            eltype(odeensemble.ics[1].q), length(odeensemble.ics[1].q)),
        p_ref::AbstractVector = zeros(
            eltype(odeensemble.ics[1].p), length(odeensemble.ics[1].p)),
        pullback_mode::Symbol = :jacobian)
    N = encoder.architecture.full_dim
    n = encoder.architecture.reduced_dim
    v_eq = odeensemble.equation.v
    f_eq = odeensemble.equation.f
    h_eq = odeensemble.equation.hamiltonian
    HRedSys(N, n, encoder, decoder, v_eq, f_eq, h_eq, odeensemble.timespan,
        odeensemble.timestep, odeensemble.ics[1];
        parameters = odeensemble.parameters[1],
        q_ref = q_ref,
        p_ref = p_ref,
        pullback_mode = pullback_mode,
        integrator = integrator)
end

# The full vector fields at the decoded point `Ψ(ξ)`, `ξ = (q̃, p̃)`, and the Jacobian `∇Ψ` at `ξ`.
# `ForwardDiff.jacobian!` gives both `Ψ(ξ)` and `∇Ψ` from the passes of the decoder in dual numbers:
# one pass where `length(ξ)` is at most ForwardDiff's chunk threshold (12 by default), and one
# pass per chunk above it.
function evaluate_vf_and_compute_∇Ψ(t, q̃::AbstractVector{Tq}, p̃::AbstractVector{Tp},
        parameters, decoder, v_full, f_full,
        cache::ReducedVFCache = ReducedVFCache(
            promote_type(Tq, Tp),
            decoder.architecture.full_dim ÷ 2,
            decoder.architecture.reduced_dim ÷ 2),
        q_ref = nothing, p_ref = nothing) where {Tq, Tp}
    T = promote_type(Tq, Tp)
    N2 = decoder.architecture.full_dim ÷ 2
    n2 = decoder.architecture.reduced_dim ÷ 2
    c = _get_vf_cache(cache, T, N2, n2)
    ξ_buf = c.ξ_buf
    v_intermediate = c.v_intermediate
    f_intermediate = c.f_intermediate
    copyto!(@view(ξ_buf[1:n2]), q̃)
    copyto!(@view(ξ_buf[(n2 + 1):(2 * n2)]), p̃)
    result = ForwardDiff.jacobian!(c.jac_result, decoder, ξ_buf)
    z = DiffResults.value(result)::Vector{T}
    _add_ref!(z, q_ref, p_ref, N2)
    q, p = @views z[1:N2], z[(N2 + 1):(2 * N2)]
    fill!(v_intermediate, zero(T))
    fill!(f_intermediate, zero(T))
    v_full(v_intermediate, t, q, p, parameters)
    f_full(f_intermediate, t, q, p, parameters)
    # Copies, so that no caller holds memory of the cache.
    copy(v_intermediate), copy(f_intermediate),
    copy(DiffResults.jacobian(result)::Matrix{T})
end

function evaluate_vf_and_vjp(t, q̃::AbstractVector{Tq}, p̃::AbstractVector{Tp},
        parameters, decoder, v_full, f_full,
        cache::ReducedVFCache, q_ref = nothing, p_ref = nothing) where {Tq, Tp}
    T = promote_type(Tq, Tp)
    N2 = decoder.architecture.full_dim ÷ 2
    n2 = decoder.architecture.reduced_dim ÷ 2
    c = _get_vf_cache(cache, T, N2, n2)
    ξ_buf = c.ξ_buf
    v_intermediate = c.v_intermediate
    f_intermediate = c.f_intermediate
    z_bar_buf = c.z_bar_buf
    copyto!(@view(ξ_buf[1:n2]), q̃)
    copyto!(@view(ξ_buf[(n2 + 1):(2 * n2)]), p̃)
    z_raw, back = Zygote.pullback(decoder, ξ_buf)
    copyto!(z_bar_buf, z_raw)
    _add_ref!(z_bar_buf, q_ref, p_ref, N2)
    q, p = @views z_bar_buf[1:N2], z_bar_buf[(N2 + 1):(2 * N2)]
    fill!(v_intermediate, zero(T))
    fill!(f_intermediate, zero(T))
    v_full(v_intermediate, t, q, p, parameters)
    f_full(f_intermediate, t, q, p, parameters)
    @inbounds for i in 1:N2
        z_bar_buf[i] = -f_intermediate[i]
        z_bar_buf[N2 + i] = v_intermediate[i]
    end
    first(back(z_bar_buf))
end

@doc raw"""
    build_v_reduced(v_full, f_full, decoder; pullback_mode = :jacobian, q_ref = nothing, p_ref = nothing, cache = ...)

Builds the reduced vector field (``q`` part) based on the full vector field for a Hamiltonian system. 

We derive the reduced vector field via the reduced Hamiltonian: ``\tilde{H} := H\circ\Psi^\mathrm{dec}``. 

We then get 
```math
\begin{aligned}
\mathbb{J}_{2n}\nabla_\xi\tilde{H} = \mathbb{J}_{2n}(\nabla\Psi^\mathrm{dec})^T\mathbb{J}_{2N}^T\mathbb{J}_{2N}\nabla_z{}H & = \mathbb{J}_{2n}(\nabla\Psi^\mathrm{dec})^T\mathbb{J}_{2N}^T \begin{pmatrix} v(z) \\ f(z) \end{pmatrix} \\ & = \begin{pmatrix} - (\nabla_p\Psi_q)^Tf(z) + (\nabla_p\Psi_p)^Tv(z) \\ (\nabla_q\Psi_q)^Tf(z) - (\nabla_q\Psi_p)^Tv(z) \end{pmatrix}.
\end{aligned}
```

`build_v_reduced` outputs the first half of the entries of this vector field.
When `pullback_mode = :jacobian` (default), `ForwardDiff.jacobian!` with `DiffResults.MutableDiffResult` computes ``\Psi^{\mathrm{dec}}(\xi)`` and ``\nabla_\xi \Psi^{\mathrm{dec}}`` in one pass.
When `pullback_mode = :vjp`, `(v(z), f(z))` is evaluated once at ``z = \Psi^{\mathrm{dec}}(\xi)`` and ``(\nabla_\xi \Psi^{\mathrm{dec}})^T \mathbb{J}_{2N}^T [v(z); f(z)]`` is computed via `Zygote.pullback(decoder, ξ)`.
The default `cache` has the element type of the decoder's parameters. A state of another floating-point element type than the cache's raises an `ArgumentError`.

!!! note "`:vjp` is not faster than `:jacobian` for small problems"
    Each of the two fields, `v_reduced` and `f_reduced`, has its own pullback, so `:vjp` runs the decoder forward and backward once per field, where `:jacobian` pays one forward pass in dual numbers per call. For `2n = 4` and a decoder of `WeightTiedSymplecticAutoencoder(2N, 4; n_blocks = 1, n_layers = 1, n_ambient_layers = 0, sympnet_upscale = 2)` in `Float64`, one call of `v_reduced` allocates 4448 B (`:jacobian`) against 27968 B (`:vjp`) for `2N = 12`, 8160 B against 29376 B for `2N = 40`, and 29728 B against 37248 B for `2N = 200`. A call takes 0.7 μs against 99 μs for `2N = 12`, 0.8 μs against 104 μs for `2N = 40`, and 2.5 μs against 106 μs for `2N = 200`. The allocation of both modes grows with `N`, and the figures are minima of warm calls in a fresh process with `--check-bounds=auto` on one machine. Use `:jacobian` unless a measurement of your own problem says otherwise.
"""
function build_v_reduced(v_full, f_full, decoder::NeuralNetwork{<:SymplecticDecoder};
        pullback_mode::Symbol = :jacobian,
        q_ref = nothing,
        p_ref = nothing,
        cache::ReducedVFCache = ReducedVFCache(
            _leaf_eltype(params(params(decoder))), decoder.architecture.full_dim ÷ 2,
            decoder.architecture.reduced_dim ÷ 2))
    _check_pullback_mode(pullback_mode)
    N2 = decoder.architecture.full_dim ÷ 2
    n2 = decoder.architecture.reduced_dim ÷ 2
    function v_reduced(
            v, t, q̃::AbstractVector{Tq}, p̃::AbstractVector{Tp}, parameters) where {Tq, Tp}
        T = promote_type(Tq, Tp)
        if pullback_mode === :vjp && (T <: AbstractFloat)
            g_ξ = evaluate_vf_and_vjp(
                t, q̃, p̃, parameters, decoder, v_full, f_full, cache, q_ref, p_ref)
            v .= @view g_ξ[(n2 + 1):(2 * n2)]
        else
            v_intermediate, f_intermediate, ∇Ψ = evaluate_vf_and_compute_∇Ψ(
                t, q̃, p̃, parameters, decoder, v_full, f_full, cache, q_ref, p_ref)
            ∇₂Ψ₁ = @view ∇Ψ[1:N2, (n2 + 1):(2 * n2)]
            ∇₂Ψ₂ = @view ∇Ψ[(N2 + 1):(2 * N2), (n2 + 1):(2 * n2)]
            v .= -∇₂Ψ₁' * f_intermediate + ∇₂Ψ₂' * v_intermediate
        end

        nothing
    end
    v_reduced
end

@doc raw"""
    build_f_reduced(v_full, f_full, decoder; pullback_mode = :jacobian, q_ref = nothing, p_ref = nothing, cache = ...)

Builds the reduced vector field (``p`` part) based on the full vector field for a Hamiltonian system. 

`build_f_reduced` outputs the second half of the entries of this vector field.

See [`build_v_reduced`](@ref) for more information.

!!! note "`:vjp` is not faster than `:jacobian` for small problems"
    Each field has its own pullback, so `pullback_mode = :vjp` allocates more and runs slower than `:jacobian` for small problems. For `2n = 4` in `Float64`, one call of `f_reduced` allocates 4272 B (`:jacobian`) against 27968 B (`:vjp`) for `2N = 12`, 7744 B against 29376 B for `2N = 40`, and 28048 B against 37248 B for `2N = 200`, and takes 0.7 μs against 99 μs, 0.8 μs against 98 μs, and 2.6 μs against 105 μs. [`build_v_reduced`](@ref) states the setup of these measurements.
"""
function build_f_reduced(v_full, f_full, decoder::NeuralNetwork{<:SymplecticDecoder};
        pullback_mode::Symbol = :jacobian,
        q_ref = nothing,
        p_ref = nothing,
        cache::ReducedVFCache = ReducedVFCache(
            _leaf_eltype(params(params(decoder))), decoder.architecture.full_dim ÷ 2,
            decoder.architecture.reduced_dim ÷ 2))
    _check_pullback_mode(pullback_mode)
    N2 = decoder.architecture.full_dim ÷ 2
    n2 = decoder.architecture.reduced_dim ÷ 2
    function f_reduced(
            f, t, q̃::AbstractVector{Tq}, p̃::AbstractVector{Tp}, parameters) where {Tq, Tp}
        T = promote_type(Tq, Tp)
        if pullback_mode === :vjp && (T <: AbstractFloat)
            g_ξ = evaluate_vf_and_vjp(
                t, q̃, p̃, parameters, decoder, v_full, f_full, cache, q_ref, p_ref)
            f .= .-(@view g_ξ[1:n2])
        else
            v_intermediate, f_intermediate, ∇Ψ = evaluate_vf_and_compute_∇Ψ(
                t, q̃, p̃, parameters, decoder, v_full, f_full, cache, q_ref, p_ref)
            ∇₁Ψ₁ = @view ∇Ψ[1:N2, 1:n2]
            ∇₁Ψ₂ = @view ∇Ψ[(N2 + 1):(2 * N2), 1:n2]
            f .= ∇₁Ψ₁' * f_intermediate - ∇₁Ψ₂' * v_intermediate
        end

        nothing
    end
    f_reduced
end

function build_h_reduced(
        h_full, decoder::NeuralNetwork{<:SymplecticDecoder};
        q_ref, p_ref)
    function h_reduced(t, q, p, params)
        dec = decoder((q = q, p = p))
        h_full(t, dec.q .+ q_ref, dec.p .+ p_ref, params)
    end
    h_reduced
end

function encode(rs::HRedSys, qp::NamedTuple{(:q, :p)})
    rs.encoder((q = qp.q .- rs.q_ref, p = qp.p .- rs.p_ref))
end

function encode(rs::HRedSys, z::AbstractVector)
    N2 = length(rs.q_ref)
    rs.encoder(vcat(@view(z[1:N2]) .- rs.q_ref, @view(z[(N2 + 1):(2 * N2)]) .- rs.p_ref))
end

function decode(rs::HRedSys, qp̃::NamedTuple{(:q, :p)})
    dec = rs.decoder(qp̃)
    (q = dec.q .+ rs.q_ref, p = dec.p .+ rs.p_ref)
end

function decode(rs::HRedSys, ξ::AbstractVector)
    z = rs.decoder(ξ)
    N2 = length(rs.q_ref)
    vcat(@view(z[1:N2]) .+ rs.q_ref, @view(z[(N2 + 1):(2 * N2)]) .+ rs.p_ref)
end

function integrate_reduced_system(rs::HRedSys)
    ics_reduced_nt = encode(rs, rs.ics)
    # convert to StateVariable format 
    ics_reduced = (q = StateVariable(ics_reduced_nt.q), p = StateVariable(ics_reduced_nt.p))
    hode = HODEProblem(rs.v_reduced, rs.f_reduced, rs.h_reduced, rs.timespan,
        rs.timestep, ics_reduced; parameters = rs.parameters)
    integrate(hode, rs.integrator)
end

function integrate_full_system(rs::HRedSys)
    hode = HODEProblem(rs.v_full, rs.f_full, rs.h_full, rs.timespan,
        rs.timestep, rs.ics; parameters = rs.parameters)
    integrate(hode, rs.integrator)
end

@doc raw"""
    reduction_error(rs)

Compute the reduction error for a [`HRedSys`](@ref).

# Arguments

If the full system and the reduced system have already been integrated, then the reduction error can be computed quicker:

```julia
reduction_error(rs, sol_full, sol_reduced)
```
This saves the cost of again integrating the respective systems.
"""
function reduction_error(rs::HRedSys, sol_full = integrate_full_system(rs),
        sol_reduced = integrate_reduced_system(rs))
    sol_full_matrices = data_tensors_from_geometric_solution(sol_full)
    sol_reduced_matrices = data_tensors_from_geometric_solution(sol_reduced)
    _norm(_diff(decode(rs, sol_reduced_matrices), sol_full_matrices)) /
    _norm(sol_full_matrices)
end

@doc raw"""
    projection_error(rs)

Compute the projection error for a [`HRedSys`](@ref).

# Arguments

If the full system has already been integrated, then the projection error can be computed quicker:

```julia
projection_error(rs, sol_full)
```
This saves the cost of again integrating the systems.
"""
function projection_error(rs::HRedSys, sol_full = integrate_full_system(rs))
    sol_full_matrices = data_tensors_from_geometric_solution(sol_full)
    _norm(_diff(decode(rs, encode(rs, sol_full_matrices)), sol_full_matrices)) /
    _norm(sol_full_matrices)
end
