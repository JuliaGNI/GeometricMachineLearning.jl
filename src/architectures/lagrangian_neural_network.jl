
const DEFAULT_LNN_NRUNS = 1000
const DEFAULT_LNN_LEARNING_RATE = .001
const DEFAULT_BATCH_SIZE = 10


struct LagrangianNeuralNetwork{AT} <: AbstractArchitecture
    dimin::Int
    width::Int
    nhidden::Int
    act::AT

    function LagrangianNeuralNetwork(dimin; width=dimin, nhidden=1, activation=tanh)
        new{typeof(activation)}(dimin, width, nhidden, activation)
    end
end

function chain(nn::LagrangianNeuralNetwork, ::LuxBackend)
    inner_layers = Tuple(
        [Lux.Dense(nn.width, nn.width, nn.act) for _ in 1:nn.nhidden]
    )

    Lux.Chain(
        Lux.Dense(nn.dimin, nn.width, nn.act),
        inner_layers...,
        Lux.Dense(nn.width, 1; use_bias=false)
    )
end


# evaluation of the Lagrangian Neural Network
(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork})(q, q̇, params = nn.params) = sum(apply(nn, [q...,q̇...], params))
(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork})(x, params = nn.params) = sum(apply(nn, x, params))

# gradient of the Lagrangian Neural Network
∇qL(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, q, q̇, params = nn.params) = Zygote.gradient(q->nn(q, q̇, params),q)[1]

# hessian of the Lagrangian Neural Network
∇∇L(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, q, q̇, params = nn.params) = Zygote.hessian(x->nn(x, params),[q...,q̇...])

∇q̇∇q̇L(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, q, q̇, params = nn.params) = ∇∇L(nn,q, q̇, params)[(1+length(q̇)):end,(1+length(q̇)):end] 

∇q∇q̇L(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, q, q̇, params = nn.params) = ∇∇L(nn,q, q̇, params)[1:length(q),(1+length(q̇)):end] 

# loss for a single datum
loss_single(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, q, q̇, qdotdot, params = nn.params) = sqeuclidean(qdotdot, inv(∇q̇∇q̇L(nn, q, q̇, params))*(∇qL(nn, q, q̇, params) - ∇q∇qdotL(nn, q, q̇, params)))

# total loss
loss(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, q, q̇, qdotdot, params = nn.params) = mapreduce(i -> loss_single(nn, q[i,:], q̇[i,:],qdotdot[i,:], params), +, eachindex(q[:,1],q̇[:,1],qdotdot[:,1]))

# loss gradient
loss_gradient(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, q, q̇, qdotdot, params = nn.params) = Zygote.gradient(p -> loss(nn, q, q̇, qdotdot, p), params)[1]

# training

function train!(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, data_q, data_q̇, target_qdotdot; ntraining = DEFAULT_LNN_NRUNS, learning_rate = DEFAULT_LNN_LEARNING_RATE, batch_size = DEFAULT_BATCH_SIZE)
    # create array to store total loss
    total_loss = zeros(ntraining)

    # convert parameters to tuple
    params_tuple = Tuple([Tuple(x) for x in nn.params])

    # do a couple learning runs
    @showprogress 1 "Training..." for j in 1:ntraining

        # gradient step
        index = rand(axes(data_q,1), batch_size)
        params_grad = loss_gradient(nn, data_q[index], data_q̇[index], target_qdotdot[index], params_tuple)

        # make gradient steps for all the model parameters W & b
        for i in eachindex(params_tuple, params_grad)
            for (p, dp) in zip(params_tuple[i], params_grad[i])
                p .-= learning_rate .* dp
            end
        end

        # total loss i.e. loss computed over all data
        total_loss[j] = loss(nn, data_q, data_q̇, target_qdotdot)
    end

    return total_loss
end
