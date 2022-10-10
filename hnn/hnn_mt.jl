using ModelingToolkit
# using Profile
using ProgressMeter

#this contains the functions for generating the training data
include("../scripts/data.jl")

#this contains the functions for generating the plots
include("../scripts/plots.jl")

#load ModelingToolkit generated functions
est   = include("../mt_fun/est.jl")
field = include("../mt_fun/field.jl")
loss  = include("../mt_fun/loss.jl")
step  = include("../mt_fun/step.jl")

#settings and common functionality
include("../mt_fun/common.jl")

#learning rate
const η = .001

#number of training runs
const runs = 1000


function training!(model, data, target, η, nruns, loss, step)
	arr_loss = zeros(nruns)

	@showprogress 1 "Training..." for j in 1:nruns
		#get batch data
		batch_data, batch_target = get_batch(data, target)

		#perform one training step
		for i in eachindex(batch_data, batch_target)
			model_grad = step(batch_data[i], batch_target[i], expand(model)...)
			for i in eachindex(model, model_grad)
				for (m, dm) in zip(model[i], model_grad[i])
					m .-= η .* dm
				end
			end
		end

		#total loss i.e. loss computed over all data
		arr_loss[j] = mapreduce(i -> loss(data[i], target[i], expand(model)...), +, eachindex(data, target))
	end

	return arr_loss
end

function train_mt_hnn(n_in, ld, η, runs, DT=Float64)
	#initialise weights
	model = (
		(W = randn(DT, ld, n_in), b = randn(DT, ld)),
		(W = randn(DT, ld, ld),   b = randn(DT, ld)),
		(W = randn(DT, 1,  ld),                    ),
	)

	#get data set
	data, target = get_data_set()

	#perform training (returns array that contains the total loss for each training step)
	total_loss = training!(model, data, target, η, runs, loss, step)

	return (model, data, target, total_loss)
end

#train network
model, data, target, total_loss = train_mt_hnn(n_in, ld, η, runs)

#time training (after warmup)
# train_mt_hnn(n_in, ld, η, 1)
# @time model, data, target, total_loss = train_mt_hnn(n_in, ld, η, runs)

#profile training
#run with julia --track-allocation=user hnn.jl
# Profile.clear()
# Profile.clear_malloc_data()
# @profile model, data, target, total_loss = train_mt_hnn(n_in, ld, η, runs)

#learned Hamiltonian & vector field
H_est(τ) = est(τ, expand(model)...)
# dH_est(τ) = field(τ, expand(model)...)

#plot results
plot_network(H, H_est, total_loss; filename="hnn_mt.png")
