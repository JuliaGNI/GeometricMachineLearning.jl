"""
There are two routines: get_initial_condition and get_initial_condition2. Their difference lies in the value for p. 
"""

using GeometricIntegrators
using HDF5

include("vector_fields.jl")
include("initial_condition.jl")

Ñ = 128
T = Float64
# params = (μ=T(.5), N=N, Δx=T(1/(N-1)))
n_params = 20
n_time_steps = 200
p_zero = false

μ_left = T(5/12)
μ_right = T(4/6)

function perform_integration(params, n_time_steps)
    tspan = (T(0),T(1))
    timestep = T((tspan[2] - tspan[1])/(n_time_steps-1))
    ics_offset = p_zero ? get_initial_condition2(params.μ, params.Ñ) : get_initial_condition(params.μ, params.Ñ)
    ics = (q=ics_offset.q.parent, p=ics_offset.p.parent)
    ode = HODEProblem(v_f_hamiltonian(params)..., parameters=params, tspan, timestep, ics)
    sol = integrate(ode, ImplicitMidpoint())
end

function perform_multiple_integration(ℙ, n_time_steps, Ñ=Ñ)
    sols = ()
    for μ in ℙ
        print("Now performing integration for μ="*string(μ)*"\n")
        params = (μ=μ, Ñ=Ñ, Δx=T(1/(Ñ-1)))
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

function generate_and_safe_data(ℙ, n_time_steps, Ñ=Ñ, n_params=n_params, file_name="snapshot_matrix.h5")
    h5open(file_name, "w") do h5
        h5["data"] = perform_multiple_integration(ℙ, n_time_steps, Ñ)
        h5["n_params"] = n_params
    end
end

function generate_and_safe_data(;n_params::Integer=n_params, n_time_steps=n_time_steps, Ñ=Ñ,file_name="snapshot_matrix.h5")
    ℙ = T(μ_left):T((μ_right-μ_left)/(n_params-1)):T(μ_right)
    generate_and_safe_data(ℙ, n_time_steps, Ñ, n_params, file_name)
end

p_zero ? generate_and_safe_data(file_name="snapshot_matrix2.h5") : generate_and_safe_data(file_name="snapshot_matrix.h5")