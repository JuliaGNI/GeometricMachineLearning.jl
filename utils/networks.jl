
#evaluate neural network
function network(τ, model)
	#first layer
	layer1 = tanh.(model[1].W * τ .+ model[1].b)

	#second layer
	layer2 = tanh.(model[2].W * layer1 .+ model[2].b)

	#third layer (linear activation)
	return model[3].W * layer2
end

