"""
ReducedSystem computes the reconstructed dynamics in the full system based on the reduced one. Optionally it can be compared to the FOM solution.
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
    projection_error
end

function ReducedSystem(N::Integer, n::Integer, encoder, decoder, full_vector_field, params, tspan, tstep, ics, projection_error; T=Float64, integrator=ImplicitMidpoint(), system_type=Symplectic())
    ReducedSystem{T, typeof(system_type)}(
        N, n, encoder, decoder, full_vector_field, build_reduced_vector_field(full_vector_field, decoder, N, n, T), integrator, params, tspan, tstep, ics, projection_error 
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

function perform_integration_reduced(rs::ReducedSystem)
    ics_reduced = rs.encoder(rs.ics)
    ode = ODEProblem(rs.reduced_vector_field, parameters=rs.params, rs.tspan, rs.tstep, ics_reduced)
    integrate(ode, rs.integrator; solver=SimpleSolvers.QuasiNewton())
end

function perform_integration_full(rs::ReducedSystem)
    ode = ODEProblem(rs.full_vector_field, parameters=rs.params, rs.tspan, rs.tstep, rs.ics)
    integrate(ode, rs.integrator; solver=SimpleSolvers.QuasiNewton())
end

function compute_reduction_error(rs::ReducedSystem)
    n_time_steps = (rs.tspan[2] - rs.tspan[1])/rs.tstep + 1
    sol_red = perform_integration_reduced(rs)
    sol_full = perform_integration_full(rs)
    sol_matrix_red = zeros(2*N, n_time_steps)
    for (t_ind,q) in zip(1:n_time_steps,sol_red.q)
        sol_matrix_red[:, n_time_steps*μ_ind+t_ind] = rs.decoder(q)
    end
    sol_matrix_full = zeros(2*N, n_time_steps)
    for (t_ind,q) in zip(1:n_time_steps,sol_full.q)
        sol_matrix_full[:, n_time_steps*μ_ind+t_ind] = q
    end
    norm(sol_matrix_red - sol_matrix_full)/norm(sol_matrix_full)
end