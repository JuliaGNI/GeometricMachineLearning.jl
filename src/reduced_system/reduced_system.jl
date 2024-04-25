@doc raw"""
ReducedSystem computes the reconstructed dynamics in the full system based on the reduced one. Optionally it can be compared to the FOM solution.

It can be called using the following constructor: `ReducedSystem(N, n; encoder, decoder, full_vector_field, reduced_vector_field, parameters, tspan, tstep, ics, projection_error)` where 
- `encoder`: a function ``\mathbb{R}^{2N}\mapsto{}\mathbb{R}^{2n}``
- `decoder`: a (differentiable) function ``\mathbb{R}^{2n}\mapsto\mathbb{R}^{2N}``
- `full_vector_field`: a (differentiable) mapping defined the same way as in GeometricIntegrators 
- `reduced_vector_field`: a (differentiable) mapping defined the same way as in GeometricIntegrators 
- `parameters`: a NamedTuple that parametrizes the vector fields (the same for full_vector_field and reduced_vector_field)
- `tspan`: a tuple `(t₀, tₗ)` that specifies start and end point of the time interval over which integration is performed. 
- `tstep`: the time step 
- `ics`: the initial condition for the big system.
- `projection_error`: the error ``||M - \mathcal{R}\circ\mathcal{P}(M)||`` where ``M`` is the snapshot matrix; ``\mathcal{P}$ and $\mathcal{R}`` are the reduction and reconstruction respectively.
"""
struct ReducedSystem{T, ET <: NeuralNetwork{<:Encoder}, DT <: NeuralNetwork{<:Decoder}, FT <: Function, RT <: Function, InT, PT, IT <: Union{AbstractArray{T}, NamedTuple{(:q, :p), Tuple{<:AbstractArray{T}, <:AbstractArray{T}}}}}
    N::Int 
    n::Int
    encoder::ET
    decoder::DT
    full_vector_field::FT
    reduced_vector_field::RT 
    integrator::InT 
    parameters::PT
    tspan::Tuple{Int, Int} 
    tstep::T
    ics::IT
end

function ReducedSystem(N::Integer, n::Integer; encoder::NeuralNetwork{<:Encoder}, decoder::NeuralNetwork{<:Decoder}, full_vector_field, parameters, tspan::Tuple, tstep::Real, ics, reduced_vector_field = build_reduced_vector_field(full_vector_field, decoder, N, n, T), integrator=ImplicitMidpoint()) 
    ReducedSystem{typeof(tstep), typeof(encoder), typeof(decoder), typeof(full_vector_field), typeof(reduced_vector_field), typeof(integrator), typeof(parameters), typeof(ics)}(N, n, encoder, decoder, full_vector_field, reduced_vector_field, integrator, parameters, tspan, tstep, ics)
end

function ReducedSystem(odeproblem::Union{ODEProblem, HODEProblem, ODEEnsemble, HODEEnsemble}; encoder::NeuralNetwork{<:Encoder}, decoder::NeuralNetwork{<:Decoder}, integrator=ImplicitMidpoint()) 
    N = encoder.architecture.full_dim 
    n = encoder.architecture.reduced_dim
    eq = odeproblem.equation
    ReducedSystem(N, n; encoder = encoder, decoder = decoder, full_vector_field = eq.v, parameters = odeproblem.parameters, tspan = odeproblem.tspan, ics = odeproblem.ics, integrator = integrator)
end

function build_reduced_vector_field(full_vector_field, decoder::SymplecticDecoder, N::Integer, n::Integer, T::DataType=Float64)
    @assert iseven(N) && iseven(n) "Full dimension and reduced dimension must be even!"
    N2 = N ÷ 2 
    n2 = n ÷ 2
    function reduced_vector_field(v, t, ξ, parameters)
        v_intermediate = zeros(T, N)
        full_vector_field(v_intermediate, t, decoder(ξ), parameters)
        v .= -SymplecticPotential(n2) * ForwardDiff.jacobian(decoder, ξ)' * SymplecticPotential(N2) * v_intermediate
    end
    reduced_vector_field
end

function perform_integration_reduced(rs::ReducedSystem)
    ics_reduced = rs.encoder(rs.ics)
    ode = ODEProblem(rs.reduced_vector_field, parameters=rs.parameters, rs.tspan, rs.tstep, ics_reduced)
    integrate(ode, rs.integrator)
end

function perform_integration_full(rs::ReducedSystem)
    ode = ODEProblem(rs.full_vector_field, parameters=rs.parameters, rs.tspan, rs.tstep, rs.ics)
    integrate(ode, rs.integrator)
end

function compute_reduction_error(rs::ReducedSystem)
    sol_full = perform_integration_full(rs)
    compute_reduction_error(rs, sol_full)
end

function compute_reduction_error(rs::ReducedSystem, sol_full)
    n_time_steps = Int(round((rs.tspan[2] - rs.tspan[1])/rs.tstep + 1))
    sol_red = perform_integration_reduced(rs)
    sol_matrix_red = zeros(rs.N, n_time_steps)
    for (t_ind,q) in zip(1:n_time_steps,sol_red.q)
        sol_matrix_red[:, t_ind] = rs.decoder(q)
    end
    sol_matrix_full = zeros(rs.N, n_time_steps)
    for (t_ind,q) in zip(1:n_time_steps,sol_full.q)
        sol_matrix_full[:, t_ind] = q
    end
    norm(sol_matrix_red - sol_matrix_full)/norm(sol_matrix_full)
end

function compute_projection_error(rs::ReducedSystem, sol_full)
    n_time_steps = Int(round((rs.tspan[2] - rs.tspan[1])/rs.tstep + 1))
    sol_matrix_full = zeros(rs.N, n_time_steps)
    for (t_ind,q) in zip(1:n_time_steps,sol_full.q)
        sol_matrix_full[:, t_ind] = q
    end
    norm(rs.decoder(rs.encoder(sol_matrix_full)) - sol_matrix_full)/norm(sol_matrix_full)
end