using Zygote

#estimate for Hamiltonian -> two inputs
#this is always a 3 layer network - maybe change that
function build_netw(Wb, n_in=2)
	#indices for Wb
	i1 = n_in*ld
	i2 = i1 + ld
	i3 = i2 + ld*ld
	i4 = i3 + ld
	i5 = i4 + ld
	
	#build neural network
	W1 = reshape(Wb[1:i1], ld, n_in)
	b1 = Wb[i1+1:i2];
	W2 = reshape(Wb[i2+1:i3], ld, ld);
	b2 = Wb[i3+1:i4];
	W3 = reshape(Wb[i4+1:i5],1,ld);
	b3 = Wb[i5+1];
        function est(τ)
                #first layer
                layer1 = Base.tanh.(W1 * τ .+ b1)
                #second layer
                layer2 = Base.tanh.(W2 * layer1 .+ b2)
                #third layer (linear activation)
                return sum(W3 * layer2 .+ b3)
        end
end


#build NN model
function model(x, n_in=2)
        y = x[1]; t = x[2];

        function loss(Wb)
                #Wb contains the weights for all the layers
                network = build_netw(Wb, n_in)
		#estimate for Hamiltonian
                est(τ) = network(τ)
                #compute vector field (field values = vals) 
                vals(κ) = [0 1; -1 0] * gradient(χ -> est(χ),κ)[1]
                #compute loss  
		loss_p(i) = sum((vals(y[1:n_in,i])-t[1:n_in,i]).^2)
		return sum([loss_p(i) for i in axes(y,2)])
	end
end

