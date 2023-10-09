@doc raw"""
ReducedSystem computes the reconstructed dynamics in the full system based on the reduced one. Optionally it can be compared to the FOM solution.

It can be called using the following constructor: ReducedSystem(N, n, encoder, decoder, full_vector_field, reduced_vector_field, params, tspan, tstep, ics, projection_error) where 
- encoder: a function $\mathbb{R}^{2N}\mapsto{}\mathbb{R}^{2n}$
- decoder: a (differentiable) function $\mathbb{R}^{2n}\mapsto\mathbb{R}^{2N}$
- full_vector_field: a (differentiable) mapping defined the same way as in GeometricIntegrators 
- reduced_vector_field: a (differentiable) mapping defined the same way as in GeometricIntegrators 
- params: a NamedTuple that parametrizes the vector fields (the same for full_vector_field and reduced_vector_field)
- tspan: a tuple (t₀, tₗ) that specifies start and end point of the time interval over which integration is performed. 
- tstep: the time step 
- ics: the initial condition for the big system.
- projection_error: the error $||M - \mathcal{R}\circ\mathcal{P}(M)||$ where $M$ is the snapshot matrix; $\mathcal{P}$ and $\mathcal{R}$ are the reduction and reconstruction respectively.
"""
struct ReducedSystem{T, ST<:SystemType} 
    N::Integer 
    n::Integer
    encoder
    decoder 
    full_vector_field
    reduced_vector_field 
    integrator 
    params
    tspan 
    tstep
    ics

    function ReducedSystem(N::Integer, n::Integer, encoder, decoder, full_vector_field, reduced_vector_field, params, tspan, tstep, ics; integrator=ImplicitMidpoint(), system_type=Symplectic(), T=Float64) 
        new{T, typeof(system_type)}(N, n, encoder, decoder, full_vector_field, reduced_vector_field, integrator, params, tspan, tstep, ics)
    end
end

function ReducedSystem(N::Integer, n::Integer, encoder, decoder, full_vector_field, params, tspan, tstep, ics; integrator=ImplicitMidpoint(), system_type=Symplectic(), T=Float64) 
    ReducedSystem{T, typeof(system_type)}(
        N, n, encoder, decoder, full_vector_field, build_reduced_vector_field(full_vector_field, decoder, N, n, T), integrator, params, tspan, tstep, ics
    )
end

function build_reduced_vector_field(full_vector_field, decoder, N, n, T=Float64)
    function reduced_vector_field(v, t, ξ, params)
        v_intermediate = zeros(T, 2*N)
        full_vector_field(v_intermediate, t, decoder(ξ), params)
        v .= -SymplecticPotential(n)*ForwardDiff.jacobian(decoder, ξ)'*SymplecticPotential(N)*v_intermediate
    end
    reduced_vector_field
end

@doc raw"""
This function is needed if we obtain a GeometricIntegrators-like vector field from an explicit vector field $V:\mathbb{R}^{2N}\to\mathbb{R}^{2N}$. 
We need this function because build_reduced_vector_field is not working in conjunction with implicit integrators.
"""
function reduced_vector_field_from_full_explicit_vector_field(full_explicit_vector_field, decoder, N::Integer, n::Integer)
    function reduced_vector_field(v, t, ξ, params)
        v .= -SymplecticPotential(n) * ForwardDiff.jacobian(decoder, ξ)' * SymplecticPotential(N) * full_explicit_vector_field(t, decoder(ξ), params)
    end
    reduced_vector_field
end

function perform_integration_reduced(rs::ReducedSystem)
    ics_reduced = rs.encoder(rs.ics)
    ode = ODEProblem(rs.reduced_vector_field, parameters=rs.params, rs.tspan, rs.tstep, ics_reduced)
    integrate(ode, rs.integrator)
end

function perform_integration_full(rs::ReducedSystem)
    ode = ODEProblem(rs.full_vector_field, parameters=rs.params, rs.tspan, rs.tstep, rs.ics)
    integrate(ode, rs.integrator)
end

function compute_reduction_error(rs::ReducedSystem)
    sol_full = perform_integration_full(rs)
    compute_reduction_error(rs, sol_full)
end

function compute_reduction_error(rs::ReducedSystem, sol_full)
    n_time_steps = Int(round((rs.tspan[2] - rs.tspan[1])/rs.tstep + 1))
    sol_red = perform_integration_reduced(rs)
    sol_matrix_red = zeros(2*rs.N, n_time_steps)
    for (t_ind,q) in zip(1:n_time_steps,sol_red.q)
        sol_matrix_red[:, t_ind] = rs.decoder(q)
    end
    sol_matrix_full = zeros(2*rs.N, n_time_steps)
    for (t_ind,q) in zip(1:n_time_steps,sol_full.q)
        sol_matrix_full[:, t_ind] = q
    end
    norm(sol_matrix_red - sol_matrix_full)/norm(sol_matrix_full)
end

function compute_projection_error(rs::ReducedSystem, sol_full)
    n_time_steps = Int(round((rs.tspan[2] - rs.tspan[1])/rs.tstep + 1))
    sol_matrix_full = zeros(2*rs.N, n_time_steps)
    for (t_ind,q) in zip(1:n_time_steps,sol_full.q)
        sol_matrix_full[:, t_ind] = q
    end
    norm(rs.decoder(rs.encoder(sol_matrix_full)) - sol_matrix_full)/norm(sol_matrix_full)
end