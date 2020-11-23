using ForwardDiff
using Zygote

#this contains the functions for building the network
include("utils.jl")

#get data set (includes dat & target)
include("plots_data/data.jl")
dat, target = get_data_set()

#layer dimension/width
ld = 5

#number of inputs/dimension of system
n_in = 2

#size of Wb: first layer + second layer + third layer
Wb_siz = (n_in*ld + ld) + (ld*ld + ld) + (ld + 1)

#initialise weights
Wb = randn(Wb_siz)

function get_batch(batch_size=10)
	#select a number of points at random (one batch)
	index = rand(axes(dat,2), batch_size)
	dat_loc = dat[1:n_in, index]
	target_loc = target[1:n_in, index]
	return((dat_loc, target_loc))
end


#learning rate
η = .001

#do a copule learning runs
runs = 1000
#array that contains the total loss for each training step
arr_loss = zeros(runs)
#total loss i.e. loss computed over all data
total_loss = model((dat,target))
for j in 1:runs
	#compute loss function for a certain batch
	local loss = model(get_batch())
	#use either Zygote (1st line) or ForwardDiff (second line)
	global Wb .-= η .* gradient(Ξ -> loss(Ξ),Wb)[1]
	#global Wb .-= η .* ForwardDiff.gradient(loss,Wb)	
	arr_loss[j] = total_loss(Wb)
end

#learned Hamiltonian & vector fields
network = build_netw(Wb)
H_est(τ) = network(τ)
#compute field
field(τ) = [[0 1; -1 0] * ForwardDiff.gradient(H_est,τ[1:n_in,i]) for i in axes(τ,2)]

##########all that follows are just diagnostics (especially plots)
#include("plots_data/plots.jl")

		
