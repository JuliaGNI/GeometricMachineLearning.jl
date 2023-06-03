
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


abstract type Hnn_training_integrator end

struct SEulerA{TD,TL} <: Hnn_training_integrator
    sqdist::TD
    loss::TL

    function SEulerA(;sqdist = sqeuclidean)

        function loss_single(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, qₙ, qₙ₊₁, pₙ, pₙ₊₁, Δt, params = nn.params)
            dH = vectorfield(nn, [qₙ₊₁...,pₙ...], params)
            sqdist(dH[1],(qₙ₊₁-qₙ)/Δt) + sqdist(dH[2],(pₙ₊₁-pₙ)/Δt)
        end

        loss(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, data::data_trajectory, index_batch, params = nn.params) = 
        mapreduce(x->loss_single(nn, data.get_q(x[1],x[2]), data.get_q(x[1],x[2]+1), data.get_p(x[1],x[2]), data.get_p(x[1],x[2]+1), data.get_Δt(), params),+, index_batch)       

        new{typeof(sqdist),typeof(loss)}(sqdist, loss)
    end
end

struct ExactIntegrator{TD,TL} <: Hnn_training_integrator
    sqdist::TD
    loss::TL

    function ExactIntegrator(;sqdist = sqeuclidean)

        loss_single(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, qₙ, pₙ, q̇ₙ, ṗₙ, params = nn.params) = sqeuclidean(vectorfield(nn, [qₙ...,pₙ...], params), [q̇ₙ..., ṗₙ...])

        loss(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, data::dataTarget{data_trajectory}, index_batch, params = nn.params) = 
        mapreduce(x->loss_single(nn, data.data.get_q(x[1],x[2]), data.data.get_p(x[1],x[2]), data.get_q̇(x[1],x[2]), data.get_ṗ(x[1],x[2]), params), +, index_batch)

        loss(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, data::dataTarget{data_sampled}, index_batch, params = nn.params) = 
        mapreduce(n->loss_single(nn, data.data.get_q(n), data.data.get_p(n), data.get_q̇(n), data.get_ṗ(n), params), +, index_batch)

        new{typeof(sqdist),typeof(loss)}(sqdist, loss)
    end
end



# loss gradient
loss_gradient(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, data, loss, index_batch, params = nn.params) = Zygote.gradient(p -> loss(nn, data, index_batch, p), params)[1]


function train!(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, m::AbstractMethodOptimiser, data::Training_data; ntraining = DEFAULT_HNN_NRUNS, hti::Hnn_training_integrator, batch_size_t)
    # create array to store total loss
    total_loss = zeros(ntraining)

    #creation of optimiser
    opt = Optimizer(m,nn.model)

    # convert parameters to tuple
    params_tuple = Tuple([Tuple(x) for x in nn.params])

    keys_1 = keys(nn.params)
    keys_2 = [keys(x) for x in values(nn.params)]

    # Learning runs
    @showprogress 1 "Training..." for j in 1:ntraining

        index_batch = get_batch(data, batch_size_t)

        params_grad = loss_gradient(nn, data, hti.loss, index_batch, params_tuple) 

        dp = NamedTuple(zip(keys_1,[NamedTuple(zip(k,x)) for (k,x) in zip(keys_2,params_grad)]))

        optimization_step!(opt, nn.model, nn.params, dp)

        total_loss[j] = 0 #hti.loss(nn, data, get_batch_multiple_trajectory(data, (data.get_nb_trajectory(), max([data.get_length_trajectory(i) for i in 1:data.get_nb_trajectory()]...))), params = nn.params)
    end

    return total_loss
end
