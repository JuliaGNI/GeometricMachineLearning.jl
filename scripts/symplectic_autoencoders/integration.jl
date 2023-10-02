using GeometricIntegrators
using HDF5

include("vector_fields.jl")
include("initial_condition.jl")

N = 256
T = Float64
# params = (μ=T(.5), N=N, Δx=T(1/(N-1)))
n_params = 8
#ℙ = T(5/12):T((5/6-5/12)/(n_params-1)):T(5/6)

function perform_integration(params, n_time_steps)
    tspan = (T(0),T(1))
    tstep = T((tspan[2] - tspan[1])/(n_time_steps-1))
    ics_offset = get_initial_condition(params.μ, params.N+2)
    ics = (q=ics_offset.q.parent, p=ics_offset.p.parent)
    ode = HODEProblem(v_f_hamiltonian(params)..., parameters=params, tspan, tstep, ics)
    sol = integrate(ode, ImplicitMidpoint())
end

function perform_multiple_integration(ℙ, n_time_steps, N=N)
    sols = ()
    for μ in ℙ
        params = (μ=T(.5), N=N, Δx=T(1/(N-1)))
        sols = (sols..., perform_integration(params, n_time_steps))
    end
    sys_dim = length(sols[1].q[0])
    sols_matrix = zeros(2*sys_dim,n_time_steps*length(ℙ))
    for (μ_ind,sol) in zip(0:(length(ℙ)-1),sols)
        for (t_ind,q,p) in zip(1:n_time_steps,sol.q,sol.p)
            sols_matrix[:, n_time_steps*μ_ind+t_ind] = vcat(q,p)
        end
    end
    sols_matrix
end 

function generate_and_safe_data(ℙ, n_time_steps, N=N, file_name="snapshot_matrix.h5")
    h5open(file_name, "w") do h5
        h5["data"] = perform_multiple_integration(ℙ, n_time_steps, N)
    end
end

function generate_and_safe_data(n_params::Integer, n_time_steps=4000, N=N, file_name="snapshot_matrix.h5")
    ℙ = T(5/12):T((5/6-5/12)/(n_params-1)):T(5/6)
    generate_and_safe_data(ℙ, n_time_steps, N, file_name)
end
