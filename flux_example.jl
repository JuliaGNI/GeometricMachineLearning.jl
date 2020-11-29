using Flux

#layer dimension
ld = 5
n_in = 2

#build model: three layers with tanh activation
model = Chain(	Dense(n_in, ld, NNlib.tanh),
		Dense(ld, ld, NNlib.tanh),
		Dense(ld, 1, NNlib.tanh))


#model parameters, i.e. W & b
Wb = params(model)


#function that computes the loss and training step
function double_grad(g,xy)
	#compute gradient of "Hamiltonian", i.e. vector field
	dgw(τ) = [0 1; -1 0] * gradient(ξ -> sum(g(ξ)), τ)[1]
	#compute loss
	#loss for single data point
	loss_sing(ξ,γ,i) = sum((dgw(ξ[1:n_in,i])-γ[1:n_in,i]).^2)
	#sum up loss
	loss(ξ,γ) = sum([loss_sing(ξ,γ,i) for i in axes(ξ,2)])	
	
	#model parameters
	mp = params(model)	
	gs = gradient(() -> loss(xy[1],xy[2]),mp)
end

#get data set
include("plots_data/data.jl")
dat, target = get_data_set()

function get_batch(batch_size=10)
	#select a number of points at random (one b	atch)
	index = rand(axes(dat,2), batch_size)
	dat_loc = dat[1:n_in, index]
	target_loc = target[1:n_in, index]
	return((dat_loc, target_loc))
end

#learning rate
η = .001

#do a copule learning runs
runs = 1000
for j in 1:runs
	#gradient step
	step = double_grad(model,get_batch()) 
	#make gradient steps for all the Wbs
	[(global Wb[i] .-= η .* step[Wb[i]]) for i in 1:length(Wb)]
end	

#compute vector field
H_est(τ) = sum(model(τ))
using ForwardDiff
field(τ) = [[0 1; -1 0] * ForwardDiff.gradient(H_est,τ[1:n_in,i]) for i in axes(τ,2)]

##########all that follows are just diagnostics (especially plots)
include("plots_data/plots.jl")


