using HDF5
using GeometricMachineLearning
using GeometricMachineLearning: QPT, QPT2
using CairoMakie
using JLD2
using NNlib: relu

include(joinpath(@__DIR__, "parametric_data_helpers.jl"))

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

# next to the script, unless GML_OUTPUT_DIR says otherwise -- an absolute path from whoever ran it
# last is no use to anybody else
path_out = joinpath(get(ENV, "GML_OUTPUT_DIR", @__DIR__), "damped_oscillator_network.jld2")


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



end


# This sets up the data loader
dl = DataLoader(turn_q_p_data_into_correct_format((q = q, p = p)))

# This sets up the neural network
arch = ForcedSympNet(2; upscaling_dimension = width, n_layers = nhidden, forcing_type = :P)
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


