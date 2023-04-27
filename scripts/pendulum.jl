using GeometricIntegrators
using Plots
using Zygote

# define Hamiltonian
H(x) = x[2]^2 / 2 + (1-cos(x[1]))
H(q, p) = H([q[1], p[1]])
H(t, q, p, params) = H(q, p)

# compute vector field
∇H(x) = Zygote.gradient(χ -> H(χ), x)[1]
dH(x) = [0 1;-1 0] * ∇H(x)

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


function pendulum_data(; tspan = (0., 100.), tstep = 0.1, q₀ = randn(1), p₀ = randn(1))
    # simulate data with geometric Integrators
    ode = HODEProblem(v, f, H, tspan, tstep, q₀, p₀)

    # sol = integrate(ode, SymplecticEulerA())
    sol = integrate(ode, SymplecticTableau(TableauExplicitEuler()))

    q = sol.q[:,1]
    p = sol.p[:,1]

    return (q, p)
end
