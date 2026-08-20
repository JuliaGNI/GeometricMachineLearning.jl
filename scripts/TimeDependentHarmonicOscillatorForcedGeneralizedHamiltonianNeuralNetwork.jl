using HDF5
using GeometricMachineLearning
using GeometricMachineLearning: QPT, QPT2, Activation, ParametricLoss, SymbolicNeuralNetwork, SymbolicPullback
using CairoMakie
using NNlib: relu

include(joinpath(@__DIR__, "parametric_data_helpers.jl"))

# PARAMETERS
omega  = 1.0                     # natural frequency of the harmonic Oscillator
Omega  = 3.5                     # frequency of the external sinusoidal forcing
F      = .9                      # amplitude of the external sinusoidal forcing   
ni_dim = 10                      # number of initial conditions per dimension (so ni_dim^2 total)
T      = 2π * 20
nt     = 1000                # number of time steps
dt     = T/nt                    # time step

# Generating the initial condition array
IC = vec( [(q=q0, p=p0) for q0 in range(-1, 1, ni_dim), p0 in range(-1, 1, ni_dim)] )

# Generating the solution array
ni = ni_dim^2
t  = collect(dt * range(0, nt, step=1))
q, p = forced_harmonic_oscillator_solution(t, IC; omega = omega, Omega = Omega, F = F)

# This sets up the data loader
dl = load_time_dependent_harmonic_oscillator_with_parametric_data_loader((q = q, p = p), t, IC)

# This sets up the neural network
width::Int = 1
nhidden::Int = 1
n_integrators::Int = 2
# sigmoid_linear_unit(x::T) where {T<:Number} = x / (T(1) + exp(-x))
arch1 = ForcedGeneralizedHamiltonianArchitecture(2; activation = tanh, width = width, nhidden = nhidden, n_integrators = n_integrators, parameters = turn_parameters_into_correct_format(t, IC)[1], forcing_type = :P)
arch2 = ForcedGeneralizedHamiltonianArchitecture(2; activation = tanh, width = width, nhidden = nhidden, n_integrators = n_integrators, parameters = turn_parameters_into_correct_format(t, IC)[1], forcing_type = :Q)
arch3 = ForcedGeneralizedHamiltonianArchitecture(2; activation = tanh, width = 2width, nhidden = nhidden, n_integrators = n_integrators, parameters = turn_parameters_into_correct_format(t, IC)[1], forcing_type = :QP)
nn1 = NeuralNetwork(arch1)
nn2 = NeuralNetwork(arch2)
nn3 = NeuralNetwork(arch3)

# This is where training starts
batch_size = 128
n_epochs = 200
batch = Batch(batch_size)
o1 = Optimizer(AdamOptimizer(), nn1)
o2 = Optimizer(AdamOptimizer(), nn2)
o3 = Optimizer(AdamOptimizer(), nn3)
loss = ParametricLoss()
_pb = SymbolicPullback(nn1, loss, turn_parameters_into_correct_format(t, IC)[1]);
_pb = SymbolicPullback(nn2, loss, turn_parameters_into_correct_format(t, IC)[1]);
_pb = SymbolicPullback(nn3, loss, turn_parameters_into_correct_format(t, IC)[1]);

function train_network()
	o1(nn1, dl, batch, n_epochs, loss, _pb)
	o2(nn2, dl, batch, n_epochs, loss, _pb)
	o3(nn3, dl, batch, n_epochs, loss, _pb)
end

loss_array = train_network()

trajectory_number = 20

# Testing the network
initial_conditions = (q = q[trajectory_number, 1], p = p[trajectory_number, 1])
n_steps = nt
trajectory = (q = zeros(1, n_steps), p = zeros(1, n_steps))
trajectory.q[:, 1] .= initial_conditions.q
trajectory.p[:, 1] .= initial_conditions.p
# note that we have to supply the parameters as a named tuple as well here:
for t_step ∈ 0:(n_steps-2)
	qp_temporary = nn3.model((q = [trajectory.q[1, t_step+1]], p = [trajectory.p[1, t_step+1]]), (t = t[t_step+1],), nn3.params)
	trajectory.q[:, t_step+2] .= qp_temporary.q
	trajectory.p[:, t_step+2] .= qp_temporary.p
end

fig = Figure()
ax = Axis(fig[1,1])
lines!(ax, trajectory.q[1,:]; label="nn")
lines!(ax, q[trajectory_number,:]; label="analytic")