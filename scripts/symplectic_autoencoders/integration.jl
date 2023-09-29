using GeometricIntegrators

include("vector_fields.jl")
include("initial_condition.jl")

N = 2048
T = Float64
params = (μ=T(.5), N=N, Δx=T(1/(N-1)))
tspan = (T(0),T(1))
n_time_steps = 4000
tstep = T((tspan[2] - tspan[1])/(n_time_steps-1))
ics_offset = get_initial_condition(params.μ, params.N+2)
ics = (q=ics_offset.q.parent, p=ics_offset.p.parent)
ode = HODEProblem(v_f_hamiltonian(params)..., parameters=params, tspan, tstep, ics)