#Default constants
const DEFAULT_SYMPNET_NRUNS = 1000
const DEFAULT_BATCH_SIZE = 10
const DEFAULT_SIZE_RESULTS = 10

# Structure
abstract type SympNet{AT,OPT} <: AbstractArchitecture end

struct LASympNet{AT,OPT} <: SympNet{AT,OPT} 
    dim::Int
    nhidden::Int
    act::AT
    opt::OPT

    function LASympNet(dim, opt; nhidden=1, activation=tanh) #default opt ?
        new{typeof(activation),typeof(opt)}(dim, width, nhidden, activation, opt)
    end

end

struct GSympNet{AT,OPT} <: SympNet{AT,OPT} 
    dim::Int
    width::Int
    nhidden::Int
    act::AT
    opt::OPT

    function GSympNet(dim, opt; width=dim, nhidden=1, activation=tanh) #default opt ?
        new{typeof(activation),typeof(opt)}(dim, width, nhidden, activation, opt)
    end
end

# Chain function
function chain(nn::GSympNet, ::LuxBackend)
    inner_layers = Tuple(
        [Gradient(nn.dim, nn.width, nn.act) for _ in 1:nn.nhidden]
    )
    Lux.Chain(
        inner_layers...
    )
end

function chain(nn::LASympNet, ::LuxBackend)
    couple_layers = []
    for _ in 1:nhidden
        push!(couple_layers,Linear(nn.dim))
        push!(couple_layers,Gradient(nn.dim, nn.dim, nn.act, full_grad = false))
    end
    
    Lux.Chain(
        couple_layers...,
        Linear(nn.dim)
    )
end

# Evaluation of the neural network

function sympnet_eval(model, q, p, params::Tuple, state)
    y = Lux.apply(model, [q...,p...], params, state)
    return y
end

function sympnet_eval(model, q, p, params::NamedTuple, state)
    y, st = Lux.apply(model, [q...,p...], params, state)
    return y
end

(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork})(q, p, params = nn.params) = sympnet_eval(nn.model, q, p, params, nn.state)


# Loss function
using StatsBase

@inline function loss(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, q, p; params = nn.params, batch_size = length(q))
    loss = 0
    Index = sample(2:length(q), batch_size, replace = false)
    for index in Index
        qp_new = nn(q[index-1], p[index-1], params)
        loss += norm(qp_new - [q[index]..., p[index]...])
    end
    return loss
end

@inline function grad_loss(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, q, p; params = nn.params, batch_size = DEFAULT_BATCH_SIZE)
    Zygote.gradient(params -> loss(nn, q, p, params=params, batch_size=batch_size), params)[1]
end


# Training

function train!(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, data_q, data_p; ntraining = DEFAULT_SYMPNET_NRUNS, batch_size = DEFAULT_BATCH_SIZE)
    
    #initialisation of optimiser


    # create array to store total loss
    total_loss = zeros(ntraining)

    # Learning runs
    @showprogress 1 "Training..." for j in 1:ntraining
        dp = grad_loss(nn, data_q, data_p; params = nn.params, batch_size = batch_size)
        apply!(nn.opt, cache, nn.model, nn.params, dp)
        total_loss[j] = loss(nn, data_q, data_p)
    end

    return total_loss
end


# Results

function Iterate_Sympnet(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, q0, p0; n_points = DEFAULT_SIZE_RESULTS)

    # Array to store the predictions
    q_learned = [zeros(lastindex(q0) for _ in 1:n_points)
    p_learned = [zeros(lastindex(p0) for _ in 1:n_points)

    # Initialisation
    q_learned[1] = q0
    p_learned[1] = p0

    #Computation of phase space
    for _ in 2:n_points
        qp_learned =  nn(q[i-1], p[i-1])
        q_learned[i] = qp_learned[1:lastindex(q0)]
        p_learned[i] = qp_learned[(1+lastindex(p0)):end]
    end

    return q_learned, p_learned
end