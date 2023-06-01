
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





abstract type training_data end

struct storing_data{T}
    Δt::Float64
    nb_trajectory::Int
    data::T
end

struct data_trajectory <: training_data
    data::storing_data
    create_empty_data::Function
    add_qp::Function
    get_Δt::Function
    get_nb_trajectory::Function
    get_length_trajectory::Function
    get_q::Function
    get_p::Function
    
    function data_trajectory(Data, Create_empty_data, Add_qp, Get_Δt,Get_nb_trajectory,Get_length_trajectory,Get_p,Get_q)
        data = training_data(Data.Δt, Data.nb_trajectory, Data.data)
        create_empty_data() = Create_empty_data()
        add_qp(i, q, p) = Add_qp(Data, i, q, p)
        get_Δt() = Get_Δt(Data)
        get_nb_trajectory() = Get_nb_trajectory(Data)
        get_length_trajectory(i) = Get_length_trajectory(Data, i)
        get_q(i,n) = Get_p(Data,i,n)
        get_p(i,n) = Get_q(Data,i,n)
        new(data,get_Δt, create_empty_data, add_qp, get_nb_trajectory, get_length_trajectory, get_q, get_p)
    end

end


function get_batch_multiple_trajectory(data::data_trajectory, batch_size = DEFAULT_BATCH_SIZE, batch_nb_trajectory = 1)
    
    Data = data.create_empty_data()
    index_trajectory = rand(1:data.get_nb_trajectory(), batch_nb_trajectory)
    for i in index_trajectory
        size_batch = max(batch_size, data.get_length_trajectory(i))
        index_qp = rand(1:data.nb_trajectory-1, size_batch÷2)
        for n in index_qp
            qₙ= data.get_q(i,n)
            pₙ= data.get_p(i,n)
            qₙ₊₁= data.get_q(i,n+1)
            pₙ₊₁= data.get_p(i,n+1)
            Data.add_qp(i,qₙ,pₙ,qₙ₊₁,pₙ₊₁)
        end
    end
    batched_data = storing_data(data.get_Δt(), data.get_nb_trajectory(), Data)
    
    return data_trajectory(batched_data, data.get_Δt, data.create_empty_data, data.add_qp, data.get_nb_trajectory, data.get_length_trajectory, data.get_q, data.get_p)
end


abstract type Hnn_training_integrator end

struct EulerA{TGB,TD,TL} <: Hnn_training_integrator
    get_batch::TGB
    sqdist::TD
    loss::TL

    function EulerA(sqdist = sqeuclidean)

        function loss(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, data, params = nn.params)

            loss = 0

            Δt = data.get_dt()

            for i in 1:data.get_nb_trajectory()
                for n in 1:data.get_length_trajectory(i)-1
                    qₙ = data.get_q(i,n)
                    qₙ₊₁ = data.get_q(i,n+1)
                    pₙ = data.get_p(i,n)
                    pₙ₊₁ = data.get_p(i,n+1)
                    dH = vectorfield(nn, [qₙ₊₁...,pₙ...], params)
                    loss += sqdist(dH[1],(qₙ₊₁-qₙ)/Δt) + sqdist(dH[2],(pₙ₊₁-pₙ)/Δt)
                end
            end
        end

        new{typeof(get_batch),typeof(dist),typeof(loss)}(get_batch_multiple_trajectory, dist, loss)
    end
end





# loss for a single datum
#loss_single(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, x, y, params = nn.params) = sqeuclidean(vectorfield(nn, x, params), y)

# total loss
#loss(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, x, y, params = nn.params) = mapreduce(i -> loss_single(nn, x[i], y[i], params), +, eachindex(x,y))

# loss gradient
loss_gradient(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, data, loss, params = nn.params) = Zygote.gradient(p -> loss(nn, data, p), params)[1]


function train!(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, m::AbstractMethodOptimiser, data; ntraining = DEFAULT_HNN_NRUNS, hti::Hnn_training_integrator, batch_size = DEFAULT_BATCH_SIZE, batch_nb_trajectory = 1)
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

        #index = rand(eachindex(data_qp), batch_size)

        params_grad = loss_gradient(nn, hti.get_batch(batch_nb_trajectory, bath_size),hti.loss, params_tuple) 

        dp = NamedTuple(zip(keys_1,[NamedTuple(zip(k,x)) for (k,x) in zip(keys_2,params_grad)]))

        optimization_step!(opt, nn.model, nn.params, dp)

        total_loss[j] = hti.loss(nn, data_qp, target)
    end

    return total_loss
end
