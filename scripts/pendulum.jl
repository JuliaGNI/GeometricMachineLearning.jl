using GeometricIntegrators

# define Hamiltonian
H(x) = x[2]^2 / 2 + (1-cos(x[1]))
H(q, p) = H([q[1], p[1]])
H(t, q, p, params) = H(q, p)

# vector field methods
function v(v, t, q, p, params)
    v[1] = p[1]
end
function f(f, t, q, p, params)
    f[1] = -sin(q[1])
end


# get data set (includes data & target)
function get_data_set(num=10, xymin=-1.2, xymax=+1.2)
	#range in which the data should be in
	rang = range(xymin, stop=xymax, length=num)

	# all combinations of (x,y) points
	data = [[x,y] for x in rang, y in rang]

	#compute the value of the vector field 
	target = dH.(data)

	return (data, target)
end


@doc raw"""
Generates data for a pendulum in 2d with optional arguments:
- `T`: the type of the data (`Float32`, `Float64`, `Float16`, etc.)
- `tspan`: default is `(0., 100.)`
- `timestep` default is `0.1`
- `q0`: default is `randn(1)`
- `p0`: default is `rand(1)`.
"""
function pendulum_data(; T = Float64, tspan = (T(0.), T(100.)), timestep = T(0.1), q0 = T.(randn(1)), p0 = T.(randn(1)))
    # simulate data with geometric Integrators
    ode = HODEProblem(v, f, H, tspan, timestep, q0, p0)

    # sol = integrate(ode, SymplecticEulerA())
    sol = integrate(ode, ImplicitMidpoint())

    n_time_steps = length(sol.t)
    q = reshape(sol.q[:,1].parent, 1, n_time_steps)
    p = reshape(sol.p[:,1].parent, 1, n_time_steps)

    # return a NamedTuple of the parent arrays.
    return (q=q, p=p)
end

function pendulum_data(ics::NamedTuple{(:q, :p), Tuple{AT, AT}}; tspan = (T(0.), T(100.)), timestep = T(0.1)) where {T, AT<:AbstractVector{T}}
    pendulum_data(; T=T, tspan=tspan, timestep=timestep, q0=ics.q, p0=ics.p)
end