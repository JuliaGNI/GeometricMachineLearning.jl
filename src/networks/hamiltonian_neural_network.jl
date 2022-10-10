
using Distances
using ForwardDiff
using ProgressMeter
using Zygote

const HamiltonianNeuralNetwork{LayersType <: Tuple} = VanillaNeuralNetwork{<: VariableWidthNetwork, LayersType}

function HamiltonianNeuralNetwork(ninput, DT = Float64; nhidden = 1, width = 5)
    # input layer
    layers = ( FeedForwardLayer(tanh, randn(DT, width, ninput), randn(DT, width)), )

    # hidden layers
    for _ in 1:nhidden
        layers = ( layers..., FeedForwardLayer(tanh, randn(DT, width, width), randn(DT, width)) )
    end

    # output layer
    layers = ( layers..., LinearFeedForwardLayer(randn(DT, 1, width), ZeroVector(DT, 1)) )

    # model
    model(x) = _applychain(layers, x)

    # vector field
    vectorfield(x) = [0 1; -1 0] * Zygote.gradient(χ -> sum(model(χ)), x)[1]

    # loss
    loss(ξ, γ) = mapreduce(i -> sqeuclidean(vectorfield(ξ[i]), γ[i]), +, eachindex(ξ,γ))

    # params
    paramtuples = ( Tuple(parameters(layer)) for layer in layers)

    params = ()

    for p in paramtuples
        params = tuplejoin(params, p)
    end

    # gradient
    grad(ξ, γ) = Zygote.gradient(() -> loss(ξ, γ), Params([params...]))[1]

    # create network
    VanillaNeuralNetwork(VariableWidthNetwork(), model, loss, grad, layers...)
end

HamiltonianNeuralNetwork(x::AbstractVector{DT}; kwargs...) where {DT} = HamiltonianNeuralNetwork(length(x), DT; kwargs...)


function apply(input::AbstractVector{DT}, network::HamiltonianNeuralNetwork) where {DT}
    apply!(zeros(DT,1), input, network)
end

# compute vector field
function vectorfield(input::AbstractVector, network::HamiltonianNeuralNetwork)
	# [0 1; -1 0] * Zygote.gradient(χ -> sum(apply(χ, network)), input)[1]
	[0 1; -1 0] * Zygote.gradient(χ -> sum(network.model(χ)), input)[1]
end

# loss for single data point
# loss_single(ξ, γ, network::HamiltonianNeuralNetwork) = sqeuclidean(vectorfield(ξ, network), γ)

# compute loss  
# loss(ξ, γ, network::HamiltonianNeuralNetwork) = mapreduce(i -> loss_single(ξ[i], γ[i], network), +, eachindex(ξ,γ))

# compute gradient of loss
# loss_gradient(Y, T, network::HamiltonianNeuralNetwork) = Zygote.gradient(net -> loss(Y, T, net), network)[1]


function train!(network::HamiltonianNeuralNetwork, data, target; ntraining = 1000, learning_rate = .001)
	arr_loss = zeros(ntraining)

	# @showprogress 1 "Training..." 
    for j in 1:ntraining
		# get batch data
		batch_data, batch_target = get_batch(data, target)

        network.loss(batch_data, batch_target)

		# compute loss function for a certain batch
		# network_grad = loss_gradient(batch_data, batch_target, network)
        # network_grad = Zygote.gradient(() -> network.loss(batch_data, batch_target), parameters(network))[1]
        network_grad = network.gradient(batch_data, batch_target)

        # update network parameters
		for i in eachindex(network.layers, network_grad)
			for (m, dm) in zip(network.layers[i], network_grad[i])
				m .-= learning_rate .* dm
			end
		end

		# total loss i.e. loss computed over all data
		arr_loss[j] = loss(data, target, network)
	end

	return arr_loss
end
