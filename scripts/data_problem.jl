using Plots
using Zygote
using  GeometricIntegrators
using LinearAlgebra


dict_problem = Dict(
    "pendulum" => (x-> x[2]^2 / 2 + (1-cos(x[1])),1),
    "Hénon_Heiles" => (x-> x[3]^2/(2) + x[4]^2/(2) + x[1]^2/2 + x[2]^2/2 + (x[1]^2*x[2]-x[2]^3/3),2)
)

# get data set (includes data & target)
function get_hamiltonian_data(nameproblem, num=10, qpmin=-1.2, qpmax=+1.2)

    #get the Hamiltonien corresponding to name_problem
    H, n_dim = dict_problem[nameproblem]
    
    # compute vector field
    ∇H(x) = Zygote.gradient(H,x)[1]

    I = Diagonal(ones(n_dim))
    Z = zeros(n_dim,n_dim)
    symplectic_matrix = [Z I;-I Z]

    dH(x) = symplectic_matrix * ∇H(x)

	#range in which the data should be in
	rang = range(qpmin, stop=qpmax, length=num)

    pre_data = collect(Iterators.product(fill(rang, 2*n_dim)...))
    data = [[x...] for x in pre_data]
    #q = [[x...] for x in pre_data, y in pre_data]
    #p = [[y...] for x in pre_data, y in pre_data]

    #data = [[q...,p...] for q in pre_data, p in pre_data]

	#compute the value of the vector field 
	target = dH.(data)

	return (data, target)
end


function get_phase_space_data(nameproblem, q0, p0, tspan = (0., 100.), tstep = 0.1)
    
    # get the Hamiltonien corresponding to name_problem   
    H_problem, n_dim = dict_problem[nameproblem] 

    H2(q, p) = H_problem([q..., p...])

    gradH(q,p) = gradient(H2,q,p)

    function v(v, t, q, p, params)
        v .= gradH(q,p)[2]
    end

    function f(f, t, q, p, params)
        f .= - gradH(q,p)[1]
    end

    h(t, q, p, params) = H2(q,p)

    # simulate data with geometric Integrators
    ode = HODEProblem(v, f, h, tspan, tstep, q0, p0)

    # sol = integrate(ode, SymplecticEulerA())
    sol = integrate(ode, SymplecticTableau(TableauExplicitEuler()))

    # results are give into a matrix where q[i,j] is the ith step of the jth component of q 
    q = hcat([sol.q[:,i] for i in 1:n_dim]...)
    p = hcat([sol.p[:,i] for i in 1:n_dim]...)

    return (q, p)
end



function compute_phase_space(H_problem, q₀, p₀, tspan2 = (0., 100.), tstep = 0.1)

    n_dim = length(q₀)

    H2(q, p) = H_problem([q..., p...])

    gradH(q,p) = gradient(H2,q,p)

    function v(v, t, q, p, params)
        v .= gradH(q,p)[2]
    end

    function f(f, t, q, p, params)
        f .= - gradH(q,p)[1]
    end

    h(t, q, p, params) = H2(q,p)

    # simulate data with geometric Integrators
    ode = HODEProblem(v, f, h, tspan2, tstep, q₀, p₀)

    # sol = integrate(ode, SymplecticEulerA())
    sol = integrate(ode, SymplecticTableau(TableauExplicitEuler()))

    q = []
    p = []
    for i in 1:(n_dim)
        push!(q,sol.q[:,i])
        push!(p,sol.p[:,i])
    end

    return (q, p)
end