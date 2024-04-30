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

function HRedSys(N::Integer, n::Integer, encoder::NeuralNetwork{<:Encoder}, decoder::NeuralNetwork{<:Decoder}, v_full, f_full, h_full, tspan::Tuple, tstep::T, ics; parameters = parameters, v_reduced = build_reduced_v(v_full, f_full, decoder), f_reduced = build_reduced_f(v_full, f_full, decoder), h_reduced = build_reduced_h(h_full, decoder), integrator=ImplicitMidpoint()) where {T <: Real}
    HRedSys{typeof(tstep), typeof(encoder), typeof(decoder), typeof(v_full), typeof(f_full), typeof(h_full), typeof(v_reduced), typeof(f_reduced), typeof(h_reduced), typeof(integrator), typeof(parameters), typeof(ics)}(N, n, encoder, decoder, v_full, f_full, h_full, v_reduced, f_reduced, h_reduced, integrator, parameters, tspan, tstep, ics)
end

function HRedSys(odeproblem::Union{HODEProblem, HODEEnsemble}, encoder::NeuralNetwork{<:Encoder}, decoder::NeuralNetwork{<:Decoder}; integrator=ImplicitMidpoint()) 
    N = encoder.architecture.full_dim 
    n = encoder.architecture.reduced_dim
    f_eq = odeproblem.equation.f
    v_eq = odeproblem.equation.v
    h_eq = odeproblem.equation.hamiltonian
    HRedSys(N, n, encoder, decoder, f_eq, v_eq, h_eq, odeproblem.tspan, odeproblem.tstep, odeproblem.ics; parameters = odeproblem.parameters, integrator = integrator)
end

@doc raw"""
Builds the reduced vector field based on the full vector field for a Hamiltonian system. We derive the reduced vector field via the reduced Hamiltonian: ``\tilde{H} := H\circ\Psi^\mathrm{dec}``. 
We then get 
```math 
\mathbb{J}_{2n}\nabla_\xi\tilde{H} = \mathbb{J}_{2n}(\nabla\Psi^\mathrm{dec})^T\mathbb{J}_{2N}^T\mathbb{J}_{2N}\nabla_z{}H = \mathbb{J}_{2n}(\nabla\Psi^\mathrm{dec})^T\mathbb{J}_{2N}^T \begin{pmatrix} v(z) \\ f(z) \end{pmatrix} = \begin{pmatrix} - (\nabla_p\Psi_q)^Tf(z) + (\nabla_p\Psi_p)^Tv(z) \\ (\nabla_q\Psi_q)^Tf(z) - (\nabla_q\Psi_p)^Tv(z) \end{pmatrix}.
```
"""
function build_reduced_v(v_full, f_full, decoder::NeuralNetwork{<:SymplecticDecoder})
    T = _eltype(decoder.params)
    N2 = decoder.architecture.full_dim ÷ 2 
    function v_reduced(v, t, q̃, p̃, parameters)
        v_intermediate = zeros(T, N2)
        f_intermediate = zeros(T, N2)
        v_full(v_intermediate, t, decoder((q = q̃, p = p̃))..., parameters)
        f_full(f_intermediate, t, decoder((q = q̃, p = p̃))..., parameters)
        ∇₂Ψ₁ = ForwardDiff.jacobian(p -> decoder(q̃, p)[1], p̃)
        ∇₂Ψ₂ = ForwardDiff.jacobian(p -> decoder(q̃, p)[2], p̃) 
        v .= -∇₂Ψ₁' * f_intermediate + ∇₂Ψ₂' * v_intermediate

        nothing
    end
    v_reduced
end

function build_reduced_f(v_full, f_full, decoder::NeuralNetwork{<:SymplecticDecoder})
    T = _eltype(decoder.params)
    N2 = decoder.architecture.full_dim ÷ 2 
    function f_reduced(f, t, q̃, p̃, parameters)
        v_intermediate = zeros(T, N2)
        f_intermediate = zeros(T, N2)
        v_full(v_intermediate, t, decoder((q = q̃, p = p̃))..., parameters)
        f_full(f_intermediate, t, decoder((q = q̃, p = p̃))..., parameters)
        ∇₁Ψ₁ = ForwardDiff.jacobian(q -> decoder(q, p̃)[1], q̃)
        ∇₁Ψ₂ = ForwardDiff.jacobian(q -> decoder(q, p̃)[2], q̃) 
        f .= ∇₁Ψ₁' * f_intermediate - ∇₁Ψ₂' * v_intermediate

        nothing
    end
    f_reduced
end

function build_reduced_h(h_full, decoder::NeuralNetwork{<:SymplecticDecoder})
    function h_reduced(t, q, p, params)
        h_full(t, decoder(q, p)..., params)
    end
    h_reduced
end

function perform_integration_reduced(rs::HRedSys)
    ics_reduced = rs.encoder(rs.ics)
    hode = HODEProblem(rs.v_reduced, rs.f_reduced,rs.h_reduced, rs.tspan, rs.tstep, ics_reduced; parameters = rs.parameters)
    integrate(hode, rs.integrator)
end

function perform_integration_full(rs::HRedSys)
    hode = HODEProblem(rs.v_full, rs.f_full, rs.h_full, rs.tspan, rs.tstep, rs.ics; parameters = rs.parameters)
    integrate(hode, rs.integrator)
end

# compute reduction error for the q part 
function compute_reduction_error(rs::HRedSys, sol_full=perform_integration_full(rs), sol_red=perform_integration_reduced(rs))
    norm(rs.decoder(sol_red.q) - sol_full.q)/norm(sol_full.q)
end

# compute projection error for the q part 
function compute_projection_error(rs::HRedSys, sol_full=perform_integration_full(rs))
    sol_full_matrices = data_matrices_from_geometric_solution(sol_full)
    _norm(_diff(rs.decoder(rs.encoder(sol_full_matrices)), sol_full_matrices)) / _norm(sol_full_matrices)
end