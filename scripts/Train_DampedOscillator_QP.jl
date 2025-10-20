using HDF5
using GeometricMachineLearning
using GeometricMachineLearning: QPT, QPT2
using CairoMakie
using JLD2
using NNlib: relu

# PARAMETERS
nu         = 0.001                     # friction force coefficient
ni_dim     = 2                      # number of initial conditions per dimension (so ni_dim^2 total)
T          = 13
nt         = 100                    # number of time steps
dt         = T/nt                    # time step
n_epochs   = 100000
n_epochs   = 3
width      = 4                      # width of the neural network
nhidden    = 3                       # number of hidden layers in the neural network
batch_size = 5000                    # the size of the batch

path_out = "D:\\RESEARCH - UTWENTE\\GFHNNs\\Damped Oscillator\\network_TEST.jld2"
#path_out = "/home/tyranowskitm/GFHNNs/DampedOscillator/OUTPUTS/network.jld2"


# Generating the initial condition array
IC = vec( [(q=q0, p=p0) for q0 in range(-1, 1, ni_dim), p0 in range(-1, 1, ni_dim)] )


# Generating the solution array
ni    = ni_dim^2
omega = sqrt(4-nu^2) / 2

q  = zeros(Float64, ni, nt+1)
p  = zeros(Float64, ni, nt+1)
t  = collect(dt*range(0,nt,step=1))

for i in 1:nt+1

	for j=1:ni
		q[j,i] =  (1/omega)*( IC[j].p + nu/2 *IC[j].q )*exp(-nu*t[i]/2)*sin(omega*t[i]) + IC[j].q*exp(-nu*t[i]/2)*cos(omega*t[i])
		p[j,i] = -(1/omega)*( IC[j].q + nu/2 *IC[j].p )*exp(-nu*t[i]/2)*sin(omega*t[i]) + IC[j].p*exp(-nu*t[i]/2)*cos(omega*t[i])
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


# This sets up the data loader
dl = DataLoader(turn_q_p_data_into_correct_format((q = q, p = p)))

# This sets up the neural network
arch = ForcedSympNet(2; upscaling_dimension = width, n_layers = nhidden, forcing_type = :QP)
#arch = ForcedSympNet(2; upscaling_dimension = width, n_layers = nhidden, activation=(x-> max(0,x)^2/2))
nn = NeuralNetwork(arch)

# This is where training starts
batch = Batch(batch_size)
o = Optimizer(AdamOptimizer(), nn)

loss_array = o(nn, dl, batch, n_epochs)


# Saving the parameters of the network
println("Saving the parameters of the neural network...")
flush(stdout)

params = GeometricMachineLearning.map_to_cpu(nn.params)

save(path_out,"parameters", params, "training loss", loss_array, "ni_dim", ni_dim, "T", T, "nt", nt, "n_epochs", n_epochs, "width", width, "nhidden", nhidden, "batch_size", batch_size, "nu", nu)

println("                            ...Done!")
flush(stdout)


