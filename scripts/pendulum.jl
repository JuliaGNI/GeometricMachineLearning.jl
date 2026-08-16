using GeometricIntegrators

# define Hamiltonian
H(x) = x[2]^2 / 2 + (1-cos(x[1]))
H(q, p) = H([q[1], p[1]])
H(t, q, p, params) = H(q, p)

# `∇H` and the symplectic gradient `dH` were dropped from this file at some point while
# `get_data_set` below kept calling `dH`, so it raised `UndefVarError`. Restored here; the gradient
# of `H(x) = x₂²/2 + (1 - cos x₁)` is written out rather than taken with Zygote, as it used to be,
# because it is two lines and the script then needs no AD to build its training data.
∇H(x) = [sin(x[1]), x[2]]
dH(x) = [0 1; -1 0] * ∇H(x)

# vector field methods
function v(v, t, q, p, params)
    v[1] = p[1]
end
function f(f, t, q, p, params)
    f[1] = -sin(q[1])
end


"""
    get_data_set(num, xymin, xymax)

A grid of `num`² points in phase space, together with the symplectic gradient at each — the
`(q, p, q̇, ṗ)` a Hamiltonian neural network trains on.

Returns a `TrainingData`. It used to return a bare `(data, target)` pair of `Matrix{Vector}`, which
`train!` has not accepted for some time.
"""
function get_data_set(num=10, xymin=-1.2, xymax=+1.2)
	#range in which the data should be in
	rang = range(xymin, stop=xymax, length=num)

	# all combinations of (x,y) points
	points = [[x, y] for x in rang for y in rang]

	#compute the value of the vector field
	derivatives = dH.(points)

	raw = (first.(points), last.(points), first.(derivatives), last.(derivatives))
	accessors = Dict(
		:shape => SampledData,
		:nb_points => Data -> length(Data[1]),
		:q => (Data, n) -> Data[1][n],
		:p => (Data, n) -> Data[2][n],
		:q̇ => (Data, n) -> Data[3][n],
		:ṗ => (Data, n) -> Data[4][n],
	)
	TrainingData(raw, accessors)
end


@doc raw"""
Generates data for a pendulum in 2d with optional arguments:
- `T`: the type of the data (`Float32`, `Float64`, `Float16`, etc.)
- `timespan`: default is `(0., 100.)`
- `timestep` default is `0.1`
- `q0`: default is `randn(1)`
- `p0`: default is `rand(1)`.
"""
function pendulum_data(; T = Float64, timespan = (T(0.), T(100.)), timestep = T(0.1), q0 = T.(randn(1)), p0 = T.(randn(1)))
    # simulate data with geometric Integrators
    ode = HODEProblem(v, f, H, timespan, timestep, q0, p0)

    # sol = integrate(ode, SymplecticEulerA())
    sol = integrate(ode, ImplicitMidpoint())

    n_time_steps = length(sol.t)
    q = reshape(sol.q[:,1].parent, 1, n_time_steps)
    p = reshape(sol.p[:,1].parent, 1, n_time_steps)

    # return a NamedTuple of the parent arrays.
    return (q=q, p=p)
end

function pendulum_data(ics::NamedTuple{(:q, :p), Tuple{AT, AT}}; timespan = (T(0.), T(100.)), timestep = T(0.1)) where {T, AT<:AbstractVector{T}}
    pendulum_data(; T=T, timespan=timespan, timestep=timestep, q0=ics.q, p0=ics.p)
end