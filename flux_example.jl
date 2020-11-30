using Flux
using ForwardDiff
using ProgressMeter

#this contains the functions for generating the training data
include("plots_data/data.jl")

#this contains the functions for generating the plots
include("plots_data/plots.jl")

#layer dimension/width
const ld = 5

#number of inputs/dimension of system
const n_in = 2

#learning rate
const η = .001

#number of training runs
const runs = 1000


function train_flux_hnn(n_in, ld, η, runs)
	#build model: three layers with tanh activation
	model = Chain( Dense(n_in, ld, NNlib.tanh),
                   Dense(ld,   ld, NNlib.tanh),
                   Dense(ld,    1, NNlib.tanh))

	#model parameters, i.e. W & b
	Wb = params(model)

	#compute gradient of "Hamiltonian", i.e. vector field
	dgw(τ) = [0 1; -1 0] * gradient(ξ -> sum(model(ξ)), τ)[1]

	#loss for single data point
	loss_sing(ξ,γ) = sum((dgw(ξ) .- γ).^2)
	
	#total loss
	loss(ξ,γ) = mapreduce(i -> loss_sing(ξ[i], γ[i]), +, eachindex(ξ,γ))

	#gradient w.r.t. model parameters
	function double_grad(ξ, γ)
		gs = gradient(() -> loss(ξ, γ), Wb)
	end

	#get data set
	data, target = get_data_set()

	#create array to store total loss
	total_loss = zeros(runs)

	#do a couple learning runs
	@showprogress 1 "Training..." for j in 1:runs
		#gradient step
		step = double_grad(get_batch(data, target)...)

		#make gradient steps for all the Wbs
		for w in Wb
			w .-= η .* step[w]
		end

		#total loss i.e. loss computed over all data
		total_loss[j] = loss(data, target)
	end

	return (model, data, target, total_loss)
end

#train network
model, data, target, total_loss = train_flux_hnn(n_in, ld, η, runs)

#learned Hamiltonian & vector field
H_est(τ) = sum(model(τ))
# dH_est(τ) = [[0 1; -1 0] * ForwardDiff.gradient(H_est,x) for x in τ]

#plot results
plot_network(H, H_est, total_loss; filename="flux.png")
