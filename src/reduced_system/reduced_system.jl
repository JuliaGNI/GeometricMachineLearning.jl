@doc raw"""
`HRedSys` computes the reconstructed dynamics in the full system based on the reduced one. Optionally it can be compared to the FOM solution.

It can be called using the following constructor: `HRedSys(N, n; encoder, decoder, v_full, f_full, v_reduced, f_reduced, parameters, tspan, tstep, ics, projection_error)` where 
- `encoder`: a function ``\mathbb{R}^{2N}\mapsto{}\mathbb{R}^{2n}``
- `decoder`: a (differentiable) function ``\mathbb{R}^{2n}\mapsto\mathbb{R}^{2N}``
- `v_full`: a (differentiable) mapping defined the same way as in GeometricIntegrators.
- `f_full`: a (differentiable) mapping defined the same way as in GeometricIntegrators.
- `v_reduced`: a (differentiable) mapping defined the same way as in GeometricIntegrators.
- `f_reduced`: a (differentiable) mapping defined the same way as in GeometricIntegrators.
- `parameters`: a NamedTuple that parametrizes the vector fields (the same for full_vector_field and reduced_vector_field)
- `tspan`: a tuple `(t₀, tₗ)` that specifies start and end point of the time interval over which integration is performed. 
- `tstep`: the time step 
- `ics`: the initial condition for the big system.
- `projection_error`: the error ``||M - \mathcal{R}\circ\mathcal{P}(M)||`` where ``M`` is the snapshot matrix; ``\mathcal{P}$ and $\mathcal{R}`` are the reduction and reconstruction respectively.
"""
struct HRedSys{
    T <: Number, 
    ET <: NeuralNetwork{<:Encoder}, 
    DT <: NeuralNetwork{<:Decoder}, 
    VFT <: Function, 
    FFT <: Function, 
    HFT <: Function, 
    VRT <: Function, 
    FRT <: Function, 
    HRT <: Function, 
    InT, # type of integrator
    PT,  # type of parameters (of the vector field/hamiltonian)
    IT <: NamedTuple{(:q, :p), Tuple{ST, ST}} where ST <: StateVariable{T}
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
    tspan::Tuple{Int, Int} 
    tstep::T
    ics::IT
end

function HRedSys(N::Integer, n::Integer, encoder::NeuralNetwork{<:Encoder}, decoder::NeuralNetwork{<:Decoder}, v_full, f_full, h_full, tspan::Tuple, tstep::T, ics; parameters = parameters, v_reduced = build_v_reduced(v_full, f_full, decoder), f_reduced = build_f_reduced(v_full, f_full, decoder), h_reduced = build_h_reduced(h_full, decoder), integrator=ImplicitMidpoint()) where {T <: Real}
    HRedSys{typeof(tstep), typeof(encoder), typeof(decoder), typeof(v_full), typeof(f_full), typeof(h_full), typeof(v_reduced), typeof(f_reduced), typeof(h_reduced), typeof(integrator), typeof(parameters), typeof(ics)}(N, n, encoder, decoder, v_full, f_full, h_full, v_reduced, f_reduced, h_reduced, integrator, parameters, tspan, tstep, ics)
end

function HRedSys(odeproblem::Union{HODEProblem, HODEEnsemble}, encoder::NeuralNetwork{<:Encoder}, decoder::NeuralNetwork{<:Decoder}; integrator=ImplicitMidpoint()) 
    N = encoder.architecture.full_dim 
    n = encoder.architecture.reduced_dim
    v_eq = odeproblem.equation.v
    f_eq = odeproblem.equation.f
    h_eq = odeproblem.equation.hamiltonian
    HRedSys(N, n, encoder, decoder, v_eq, f_eq, h_eq, odeproblem.tspan, odeproblem.tstep, odeproblem.ics; parameters = odeproblem.parameters, integrator = integrator)
end

# this is much more expensive than it has to be and is due to a problem with nested derivatives in ForwardDiff (should not be necessary to do this twice!)
function evaluate_vf_and_compute_∇Ψ(t, q̃::AbstractVector{T}, p̃::AbstractVector{T}, parameters, decoder, v_full, f_full) where T
    N2 = decoder.architecture.full_dim ÷ 2
    v_intermediate = zeros(T, N2)
    f_intermediate = zeros(T, N2)
    decoded_nt = decoder((q = q̃, p = p̃))
    v_full(v_intermediate, t, decoded_nt..., parameters)
    f_full(f_intermediate, t, decoded_nt..., parameters)
    ∇Ψ = ForwardDiff.jacobian(qp -> decoder(qp), vcat(q̃, p̃)) 
    v_intermediate, f_intermediate, ∇Ψ 
end

@doc raw"""
Builds the reduced vector field based on the full vector field for a Hamiltonian system. We derive the reduced vector field via the reduced Hamiltonian: ``\tilde{H} := H\circ\Psi^\mathrm{dec}``. 
We then get 
```math 
\mathbb{J}_{2n}\nabla_\xi\tilde{H} = \mathbb{J}_{2n}(\nabla\Psi^\mathrm{dec})^T\mathbb{J}_{2N}^T\mathbb{J}_{2N}\nabla_z{}H = \mathbb{J}_{2n}(\nabla\Psi^\mathrm{dec})^T\mathbb{J}_{2N}^T \begin{pmatrix} v(z) \\ f(z) \end{pmatrix} = \begin{pmatrix} - (\nabla_p\Psi_q)^Tf(z) + (\nabla_p\Psi_p)^Tv(z) \\ (\nabla_q\Psi_q)^Tf(z) - (\nabla_q\Psi_p)^Tv(z) \end{pmatrix}.
```
"""
function build_v_reduced(v_full, f_full, decoder::NeuralNetwork{<:SymplecticDecoder})
    N2 = decoder.architecture.full_dim ÷ 2 
    n2 = decoder.architecture.reduced_dim ÷ 2
    function v_reduced(v, t, q̃::AbstractVector{T}, p̃::AbstractVector{T}, parameters) where T
        v_intermediate, f_intermediate, ∇Ψ = evaluate_vf_and_compute_∇Ψ(t, q̃, p̃, parameters, decoder, v_full, f_full)
        ∇₂Ψ₁ = @view ∇Ψ[1:N2, (n2 + 1):(2 * n2)]
        ∇₂Ψ₂ = @view ∇Ψ[(N2 + 1):(2 * N2), (n2 + 1):(2 * n2)]
        v .= -∇₂Ψ₁' * f_intermediate + ∇₂Ψ₂' * v_intermediate

        nothing
    end
    v_reduced
end

function build_f_reduced(v_full, f_full, decoder::NeuralNetwork{<:SymplecticDecoder})
    N2 = decoder.architecture.full_dim ÷ 2 
    n2 = decoder.architecture.reduced_dim ÷ 2
    function f_reduced(f, t, q̃::AbstractVector{T}, p̃::AbstractVector{T}, parameters) where T 
        v_intermediate, f_intermediate, ∇Ψ = evaluate_vf_and_compute_∇Ψ(t, q̃, p̃, parameters, decoder, v_full, f_full)
        ∇₁Ψ₁ = @view ∇Ψ[1:N2, 1:n2]
        ∇₁Ψ₂ = @view ∇Ψ[(N2 + 1):(2 * N2), 1:n2]
        f .= ∇₁Ψ₁' * f_intermediate - ∇₁Ψ₂' * v_intermediate

        nothing
    end
    f_reduced
end

function build_h_reduced(h_full, decoder::NeuralNetwork{<:SymplecticDecoder})
    function h_reduced(t, q, p, params)
        h_full(t, decoder(q, p)..., params)
    end
    h_reduced
end

function integrate_reduced_system(rs::HRedSys)
    ics_reduced_nt = rs.encoder(rs.ics)
    # convert to StateVariable format 
    ics_reduced = (q = StateVariable(ics_reduced_nt.q), p = StateVariable(ics_reduced_nt.p))
    hode = HODEProblem(rs.v_reduced, rs.f_reduced, rs.h_reduced, rs.tspan, rs.tstep, ics_reduced; parameters = rs.parameters)
    integrate(hode, rs.integrator)
end

function integrate_full_system(rs::HRedSys)
    hode = HODEProblem(rs.v_full, rs.f_full, rs.h_full, rs.tspan, rs.tstep, rs.ics; parameters = rs.parameters)
    integrate(hode, rs.integrator)
end

# compute reduction error for the q part 
function reduction_error(rs::HRedSys, sol_full=integrate_full_system(rs), sol_reduced=integrate_reduced_system(rs))
    sol_full_matrices = data_matrices_from_geometric_solution(sol_full)
    sol_reduced_matrices = data_matrices_from_geometric_solution(sol_reduced)
    _norm(_diff(rs.decoder(sol_reduced_matrices), sol_full_matrices)) / _norm(sol_full_matrices)
end

# compute projection error for the q part 
function projection_error(rs::HRedSys, sol_full=integrate_full_system(rs))
    sol_full_matrices = data_matrices_from_geometric_solution(sol_full)
    _norm(_diff(rs.decoder(rs.encoder(sol_full_matrices)), sol_full_matrices)) / _norm(sol_full_matrices)
end