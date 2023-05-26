
const DEFAULT_HNN_NRUNS = 1000
const DEFAULT_BATCH_SIZE = 10


struct HamiltonianNeuralNetwork{AT} <: AbstractArchitecture
    dimin::Int
    width::Int
    nhidden::Int
    act::AT

    function HamiltonianNeuralNetwork(dimin; width=dimin, nhidden=1, activation=tanh)
        new{typeof(activation)}(dimin, width, nhidden, activation)
    end
end

function chain(nn::HamiltonianNeuralNetwork, ::LuxBackend)
    inner_layers = Tuple(
        [Lux.Dense(nn.width, nn.width, nn.act) for _ in 1:nn.nhidden]
    )

    Lux.Chain(
        Lux.Dense(nn.dimin, nn.width, nn.act),
        inner_layers...,
        Lux.Dense(nn.width, 1; use_bias=false)
    )
end


# evaulation of the Hamiltonian Neural Network
(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork})(x, params = nn.params) = sum(apply(nn, x, params))

# gradient of the Hamiltonian Neural Network
gradient(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, x, params = nn.params) = Zygote.gradient(ξ -> nn(ξ, params), x)[1]

# vector field of the Hamiltonian Neural Network
vectorfield(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, x, params = nn.params) = [0 1; -1 0] * gradient(nn, x, params)

# loss for a single datum
loss_single(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, x, y, params = nn.params) = sqeuclidean(vectorfield(nn, x, params), y)

# total loss
loss(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, x, y, params = nn.params) = mapreduce(i -> loss_single(nn, x[i], y[i], params), +, eachindex(x,y))

# loss gradient
loss_gradient(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, x, y, params = nn.params) = Zygote.gradient(p -> loss(nn, x, y, p), params)[1]



function train!(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, m::AbstractMethodOptimiser, data_qp, target; ntraining = DEFAULT_HNN_NRUNS, batch_size = DEFAULT_BATCH_SIZE)
    # create array to store total loss
    total_loss = zeros(ntraining)

    #creation of optimiser
    opt = Optimizer(m,nn.model)

    # convert parameters to tuple
    params_tuple = Tuple([Tuple(x) for x in nn.params])

    keys_1 = keys(nn.params)
    keys_2 = [keys(x) for x in values(nn.params)]

    learning_rate = 0.01
    # Learning runs
    @showprogress 1 "Training..." for j in 1:ntraining

        index = rand(eachindex(data_qp), batch_size)
        params_grad = loss_gradient(nn, data_qp[index], target[index], params_tuple) #loss_gradient(nn, get_batch(data_qp, target)..., params_tuple)

        dp = NamedTuple(zip(keys_1,[NamedTuple(zip(k,x)) for (k,x) in zip(keys_2,params_grad)]))

        optimization_step!(opt, nn.model, nn.params, dp)

        #=
        # make gradient steps for all the model parameters W & b
        for i in eachindex(params_tuple, params_grad)
            for (p, dp) in zip(params_tuple[i], params_grad[i])
                p .-= learning_rate .* dp
            end
        end
        =#
        # gradient step
        #index = rand(eachindex(data_qp), batch_size)
        #dp = loss_gradient(nn, data_qp[index], target[index], params_tuple)

        #optimization_step!(opt, nn.model, nn.params, dp)
        total_loss[j] = loss(nn, data_qp, target)
    end

    return total_loss
end
