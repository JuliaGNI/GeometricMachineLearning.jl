using HDF5
using GeometricMachineLearning
using GeometricMachineLearning: QPT, QPT2
using CairoMakie
using NNlib: relu

# PARAMETERS
omega  = 1.0                     # natural frequency of the harmonic Oscillator
Omega  = 3.5                     # frequency of the external sinusoidal forcing
F      = .9                      # amplitude of the external sinusoidal forcing   
ni_dim = 10                      # number of initial conditions per dimension (so ni_dim^2 total)
T      = 2π
nt     = 500                    # number of time steps
dt     = T/nt                    # time step

# Generating the initial condition array
IC = vec( [(q=q0, p=p0) for q0 in range(-1, 1, ni_dim), p0 in range(-1, 1, ni_dim)] )

# Generating the solution array
ni = ni_dim^2
q  = zeros(Float64, ni, nt+1)
p  = zeros(Float64, ni, nt+1)
t  = collect(dt * range(0, nt, step=1))

for i in 1:nt+1
	for j=1:ni
		q[j,i] =  ( IC[j].p - Omega*F/(omega^2-Omega^2) )/ omega *sin(omega*t[i]) + IC[j].q*cos(omega*t[i]) + F/(omega^2-Omega^2)*sin(Omega*t[i])
		p[j,i] = -omega^2*IC[j].q*sin(omega*t[i]) + ( IC[j].p - Omega*F/(omega^2-Omega^2) )*cos(omega*t[i]) + Omega*F/(omega^2-Omega^2)*cos(Omega*t[i])
		# q[j,i] =  ( IC[j].p - Omega*F/(omega^2-Omega^2) )/ omega *exp(-omega*t[i]) - IC[j].q*exp(-omega*t[i]) + F/(omega^2-Omega^2)*exp(-Omega*t[i])
		# p[j,i] = -omega^2*IC[j].q*exp(-omega*t[i]) + ( IC[j].p + Omega*F/(omega^2-Omega^2) )*exp(-omega*t[i]) - Omega*F/(omega^2-Omega^2)*exp(-Omega*t[i])
	end

end

@doc raw"""
Turn a `NamedTuple` of ``(q,p)`` data into two tensors of the correct format.

This is the tricky part as the structure of the input array(s) needs to conform with the structure of the parameters.

Here the data are rearranged in an array of size ``(n, 2, t_f - 1)`` where ``[t_0, t_1, \ldots, t_f]`` is the vector storing the time steps.

If we deal with different initial conditions as well, we still put everything into the third (parameter) axis.

# Example

```jldoctest
using GeometricMachineLearning

q = [1. 2. 3.; 4. 5. 6.]
p = [1.5 2.5 3.5; 4.5 5.5 6.5]
qp = (q = q, p = p)
turn_q_p_data_into_correct_format(qp)

# output

(q = [1.0 2.0; 4.0 5.0;;; 2.0 3.0; 5.0 6.0], p = [1.5 2.5; 4.5 5.5;;; 2.5 3.5; 5.5 6.5])
```
"""
function turn_q_p_data_into_correct_format(qp::QPT2{T, 2}) where {T}
	number_of_time_steps = size(qp.q, 2) - 1 # not counting t₀
	number_of_initial_conditions = size(qp.q, 1)
	q_array = zeros(T, 1, 2, number_of_time_steps * number_of_initial_conditions)
	p_array = zeros(T, 1, 2, number_of_time_steps * number_of_initial_conditions)
	for initial_condition_index ∈ 0:(number_of_initial_conditions - 1)
		for time_index ∈ 1:number_of_time_steps
			q_array[:, 1, initial_condition_index * number_of_time_steps + time_index] .= qp.q[initial_condition_index + 1, time_index]
			q_array[:, 2, initial_condition_index * number_of_time_steps + time_index] .= qp.q[initial_condition_index + 1, time_index + 1]
			p_array[:, 1, initial_condition_index * number_of_time_steps + time_index] .= qp.p[initial_condition_index + 1, time_index]
			p_array[:, 2, initial_condition_index * number_of_time_steps + time_index] .= qp.p[initial_condition_index + 1, time_index + 1]
		end
	end
	(q = q_array, p = p_array)
end

# SAVING TO FILE

# h5 = h5open(path, "w")
# write(h5, "q", q)
# write(h5, "p", p)
# write(h5, "t", t)
# 
# attrs(h5)["ni"] = ni
# attrs(h5)["nt"] = nt
# attrs(h5)["dt"] = dt
# 
# close(h5)

# This sets up the data loader
dl = DataLoader(turn_q_p_data_into_correct_format((q = q, p = p)))

# This sets up the neural network
width::Int = 5
nhidden::Int = 4
activation = relu
arch = ForcedSympNet(2; upscaling_dimension = width, 
                        n_layers = nhidden, 
                        activation = activation)
nn = NeuralNetwork(arch)

# This is where training starts
function train_network(batch_size::Integer=5000, method=AdamOptimizer(), n_epochs=1000)
	batch = Batch(batch_size)
	o = Optimizer(method, nn)
	o(nn, dl, batch, n_epochs)
end

loss_array = train_network()

trajectory_number = 20

# Testing the network
initial_conditions = (q = q[trajectory_number, 1], p = p[trajectory_number, 1])
n_steps = nt
trajectory = (q = zeros(1, n_steps), p = zeros(1, n_steps))
trajectory.q[:, 1] .= initial_conditions.q
trajectory.p[:, 1] .= initial_conditions.p
for t_step ∈ 0:(n_steps-2)
	qp_temporary = nn.model((q = [trajectory.q[1, t_step+1]], p = [trajectory.p[1, t_step+1]]), nn.params)
	trajectory.q[:, t_step+2] .= qp_temporary.q
	trajectory.p[:, t_step+2] .= qp_temporary.p
end

fig = Figure()
ax = Axis(fig[1,1])
lines!(ax, trajectory.q[1,:]; label="nn")
lines!(ax, q[trajectory_number,:]; label="analytic")