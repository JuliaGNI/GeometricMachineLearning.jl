using Plots
using Zygote
#using  GeometricIntegrators
using LinearAlgebra
using GeometricMachineLearning


#################################################################################################
# Hamiltomian Dictionary

dict_problem_H= Dict(
    :pendulum => (x-> x[2]^2 / 2 + (1-cos(x[1])),1),
    :Hénon_Heiles => (x-> x[3]^2/(2) + x[4]^2/(2) + x[1]^2/2 + x[2]^2/2 + (x[1]^2*x[2]-x[2]^3/3),2)
    #:three_body_system =>(,)
)

# Lagrangian DictionaryHH

dict_problem_L = Dict(
    :pendulum => (x-> x[2]^2 / 2 - (1-cos(x[1])),1),
)



###########################################################################
# get data of a problem for HNN (results is (data = (q,p), target = (q̇,ṗ)))

function get_HNN_data(nameproblem, num=10, qpmin=-1.2, qpmax=+1.2)

    #get the Hamiltonien corresponding to name_problem
    H, n_dim = dict_problem_H[nameproblem]
    
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

	#compute the value of the vector field 
	target = dH.(data)

	return (data, target)
end


###############################################################################
# get data of a problem for LNN (results is (data = (q,q̇), target = (qdotdot)))

function get_LNN_data(nameproblem, num=10, qq̇min=-1.2, qq̇max=+1.2)

    #get the Hamiltonien corresponding to name_problem
    L, n_dim = dict_problem_L[nameproblem]
    
    #compute gradient and hessian
    ∇qL(x) = Zygote.gradient(x->L(x),x)[1][1:(length(x)÷2)]
    ∇∇L(x) = Zygote.hessian(L,x)
    ∇q̇∇q̇L(x) = ∇∇L(x)[(1+length(x)÷2):end,(1+length(x)÷2):end] 
    ∇q∇q̇L(x) = ∇∇L(x)[1:(length(x)÷2),(1+length(x)÷2):end] 

    #compute qdotdot from Lagrangian formulation
    Qdotdot(q,q̇) = inv(∇q̇∇q̇L([q...,q̇...]))*(∇qL([q...,q̇...]) - ∇q∇q̇L([q...,q̇...])*q̇)

    #range in which the data should be in
	rang = range(qq̇min, stop=qq̇max, length=num)

    pre_data = collect(Iterators.product(fill(rang, 2*n_dim)...))
    data = [[[x[1:n_dim]...],[x[(1+n_dim):2*n_dim]...]] for x in pre_data]

    #compute target associate to data
    target = [Qdotdot(qq̇[1],qq̇[2]) for qq̇ in data]

	return (data, target)
end


########################################################################################
# get data of a problem for Sypmnet (results is (data = target =(q,p))) from Hamiltonian

function get_phase_space_data(nameproblem, q₀, p₀, tspan = (0., 100.), tstep = 0.1)
    
    # get the Hamiltonien corresponding to name_problem   
    H_problem, n_dim = dict_problem_H[nameproblem] 

    q,p = compute_phase_space(H_problem, q₀, p₀, tspan, tstep)

    return (q, p)
end


function get_phase_space_multiple_trajectoy(nameproblem; singlematrix = true, n_trajectory = 1, n_points = 10, tstep = 0.1, qmin = -0.2, pmin = -0.2, qmax = 0.2, pmax = 0.2)

    # get the Hamiltonien corresponding to name_problem   
    H_problem, n_dim = dict_problem_H[nameproblem] 

    #define timespan
    tspan=(0.,n_points*tstep)

    #compute phase space for each trajectory staring from a random point
    trajectory_q = [zeros(n_points+1,n_dim) for _ in 1:n_trajectory]
    trajectory_p = [zeros(n_points+1,n_dim) for _ in 1:n_trajectory]
    for i in 1:n_trajectory
        q₀ = [rand()*(qmax-qmin)+qmin for _ in 1:n_dim]
        p₀ = [rand()*(pmax-pmin)+pmin for _ in 1:n_dim]
        trajectory_q[i],trajectory_p[i] = compute_phase_space(H_problem, q₀, p₀, tspan, tstep)
    end

    if singlematrix
        data_q = vcat(trajectory_q...)
        data_p = vcat(trajectory_p...)
        return data_q,data_p
    else 
        return trajectory_q,trajectory_p
    end
end

###############################################################################
# compute phase space from the Hamiltonian

function compute_phase_space(H_problem, q₀, p₀, tspan = (0., 100.), tstep = 0.1)

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
    ode = HODEProblem(v, f, h, tspan, tstep, q₀, p₀)

    #return sol = integrate(ode, SymplecticEulerA())
    return sol = integrate(ode, SymplecticTableau(TableauExplicitEuler()))

    # results are give into a matrix where q[i,j] is the ith step of the jth component of q 
    q = hcat([sol.q[:,i] for i in 1:n_dim]...)
    p = hcat([sol.p[:,i] for i in 1:n_dim]...)

    return (q, p)
end


###############################################################################
# get structure data for multiple trajectory


struct storing_data{T}
    Δt::Float64
    nb_trajectory::Int
    data::T
end


function get_multiple_trajectory_structure(nameproblem; n_trajectory = 1, n_points = 10, tstep = 0.1, qmin = -1.2, pmin = -1.2, qmax = 1.2, pmax = 1.2)

    
    # get the Hamiltonien corresponding to name_problem   
    H_problem, n_dim = dict_problem_H[nameproblem] 

    #define timespan
    tspan=(0.,n_points*tstep)
    
    #compute phase space for each trajectory staring from a random point
    pre_data = NamedTuple()
    
    for i in 1:n_trajectory

        q₀ = [rand()*(qmax-qmin)+qmin for _ in 1:n_dim]
        p₀ = [rand()*(pmax-pmin)+pmin for _ in 1:n_dim]
        q, p = compute_phase_space(H_problem, q₀, p₀, tspan, tstep)

        Data = [(q[n], p[n]) for n in 1:size(q,1)]


        nt = NamedTuple{(Symbol("Trajectory_"*string(i)),)}(((data = Data, len = n_points+1),))

        pre_data = merge(pre_data,nt)
    end
    
    data = storing_data(tstep, n_trajectory, pre_data)



    return data #data_trajectory(data, Get_nb_trajectory, Get_length_trajectory, Get_q, Get_p, Get_Δt)
end

function get_multiple_trajectory_structure_with_target(nameproblem; n_trajectory = 1, n_points = 10, tstep = 0.1, qmin = -0.2, pmin = -0.2, qmax = 0.2, pmax = 0.2)

    
    # get the Hamiltonien corresponding to name_problem   
    H_problem, n_dim = dict_problem_H[nameproblem] 
    
    # compute vector field
    ∇H(x) = Zygote.gradient(H,x)[1]

    I = Diagonal(ones(n_dim))
    Z = zeros(n_dim,n_dim)
    symplectic_matrix = [Z I;-I Z]

    dH(x) = symplectic_matrix * ∇H(x)

    #define timespan
    tspan=(0.,n_points*tstep)
    
    #compute phase space for each trajectory staring from a random point
    pre_data = NamedTuple()
    Target = NamedTuple()
    
    for i in 1:n_trajectory

        q₀ = [rand()*(qmax-qmin)+qmin for _ in 1:n_dim]
        p₀ = [rand()*(pmax-pmin)+pmin for _ in 1:n_dim]
        q, p = compute_phase_space(H_problem, q₀, p₀, tspan, tstep)

        Data = [(q[n], p[n]) for n in 1:size(q,1)]
        data_calc = [[q[n]..., p[n]...] for n in 1:size(q,1)]

        nt = NamedTuple{(Symbol("Trajectory_"*string(i)),)}(((data = Data, len = n_points+1),))

        pre_data = merge(pre_data,nt)

        calc = dH.(data_calc)
        pre_target = [(calc[n][1:n_dim], calc[n][1+n_dim:end]) for n in 1:size(q,1)]
        ntt = NamedTuple{(Symbol("Trajectory_"*string(i)),)}(((target = pre_target, len = n_points+1),))
        Target = merge(Target,ntt)

    end
    
    data = storing_data(tstep, n_trajectory, pre_data)

    Get_Δt(Data) = Data.Δt
    Get_nb_trajectory(Data) = Data.nb_trajectory
    Get_length_trajectory(Data, i) = Data.data[Symbol("Trajectory_"*string(i))][:len]
    Get_q(Data, i, n) = Data.data[Symbol("Trajectory_"*string(i))][:data][n][1]
    Get_p(Data, i, n) = Data.data[Symbol("Trajectory_"*string(i))][:data][n][2]

    Get_q̇(Target,i,n)= Target[Symbol("Trajectory_"*string(i))][:target][n][1][1]
    Get_ṗ(Target,i,n)= Target[Symbol("Trajectory_"*string(i))][:target][n][2][1]

    return dataTarget(data_trajectory(data, Get_nb_trajectory, Get_length_trajectory, Get_q, Get_p, Get_Δt), Target, Get_q̇, Get_ṗ)
end


function get_multiple_trajectory_structure_Lagrangian(nameproblem; n_trajectory = 1, n_points = 10, tstep = 0.1, qmin = -0.2, pmin = -0.2, qmax = 0.2, pmax = 0.2)

    
    # get the Hamiltonien corresponding to name_problem   
    H_problem, n_dim = dict_problem_L[nameproblem] 

    #define timespan
    tspan=(0.,n_points*tstep)
    
    #compute phase space for each trajectory staring from a random point
    pre_data = []
    push!(pre_data,[tstep])
    push!(pre_data,[n_trajectory])
    push!(pre_data, [n_points+1])
    
    
    for i in 1:n_trajectory

        q₀ = [rand()*(qmax-qmin)+qmin for _ in 1:n_dim]
        p₀ = [rand()*(pmax-pmin)+pmin for _ in 1:n_dim]
        q, p = compute_phase_space(H_problem, q₀, p₀, tspan, tstep)

        Data = [q[n] for n in 1:size(q,1)]

        pre_data = push!(pre_data,Data)
    end

    return pre_data 
end