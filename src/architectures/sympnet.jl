#Default constants
const DEFAULT_SYMPNET_NRUNS = 1000
const DEFAULT_BATCH_SIZE = 10
const DEFAULT_SIZE_RESULTS = 10

# Structure
abstract type SympNet{AT} <: AbstractArchitecture end

struct LASympNet{AT,T1,T2,T3} <: SympNet{AT} 
    dim::Int
    width::Int
    nhidden::Int
    act::AT
    init_uplow_linear::Vector{Bool}
    init_uplow_act::Vector{Bool}
    init_sym_matrices::T1
    init_bias::T2
    init_weight::T3

    function LASympNet(dim; width=9, nhidden=0, activation=tanh, init_uplow_linear=[true,false], init_uplow_act=[true,false],init_sym_matrices=Lux.glorot_uniform, init_bias=Lux.zeros32, init_weight=Lux.glorot_uniform) 
        new{typeof(activation),typeof(init_sym_matrices),typeof(init_bias),typeof(init_weight)}(dim, min(width,9), nhidden, activation, init_uplow_linear, init_uplow_act, init_sym_matrices, init_bias, init_weight)
    end

end

struct GSympNet{AT,T1,T2,T3} <: SympNet{AT} 
    dim::Int
    width::Int
    nhidden::Int
    act::AT
    init_uplow::Vector{Bool}
    init_weight::T1
    init_bias::T2
    init_scale::T3

    function GSympNet(dim; width=dim, nhidden=1, activation=tanh, init_uplow=[true,false], init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32, init_scale=Lux.glorot_uniform) 
        new{typeof(activation),typeof(init_weight),typeof(init_bias),typeof(init_scale)}(dim, width, nhidden, activation, init_uplow, init_weight, init_bias, init_scale)
    end
end

# Chain function
function chain(nn::GSympNet, ::LuxBackend)
    inner_layers = Tuple(
        [Gradient(nn.dim, nn.width, nn.act, change_q = nn.init_uplow[Int64((i-1)%length(nn.init_uplow)+1)], init_weight=nn.init_weight, init_bias=nn.init_bias, init_scale=nn.init_scale) for i in 1:nn.nhidden]
    )
    Lux.Chain(
        inner_layers...
    )
end

function chain(nn::LASympNet, ::LuxBackend)
    couple_layers = []
    for i in 1:nn.nhidden
        for j in 1:nn.width
            push!(couple_layers, Linear(nn.dim, change_q = nn.init_uplow_linear[Int64((i-1+j-1)%length(nn.init_uplow_linear)+1)], bias=(j==nn.width), init_weight=nn.init_sym_matrices, init_bias=nn.init_bias))
        end
        push!(couple_layers,Gradient(nn.dim, nn.dim, nn.act, full_grad = false, change_q = nn.init_uplow_act[Int64((i-1)%length(nn.init_uplow_act)+1)], init_weight=nn.init_weight))
    end

    for j in 1:nn.width
        push!(couple_layers, Linear(nn.dim, change_q = nn.init_uplow_linear[Int64((nn.nhidden+1-1+j-1)%length(nn.init_uplow_linear)+1)], bias=(j==nn.width), init_weight=nn.init_sym_matrices, init_bias=nn.init_bias))
    end
    
    Lux.Chain(
        couple_layers...
    )
end

# Evaluation of the neural network

(nn::LuxNeuralNetwork{<:SympNet})(q, p, params = nn.params) = apply(nn, [q...,p...],params)

abstract type SympNetIntegrator end

struct BaseIntegrator{TD,TL,TA}<:SympNetIntegrator
    sqdist::TD
    loss::TL
    assert::TA
    min_length_trajectory::Int

    function BaseIntegrator(;sqdist = sqeuclidean)

        min_length_trajectory = 2

        function loss_single(nn::LuxNeuralNetwork{<:SympNet}, qₙ, pₙ, qₙ₊₁, pₙ₊₁, params = nn.params)
            sqdist(nn(qₙ,pₙ,params),[qₙ₊₁...,pₙ₊₁])
        end

        loss(nn::LuxNeuralNetwork{<:SympNet}, datat::data_trajectory, index_batch = get_batch(datat), params = nn.params) =
        mapreduce(x->loss_single(nn, datat.get_data[:q](x[1],x[2]), datat.get_data[:p](x[1],x[2]), datat.get_data[:q](x[1],x[2]+1), datat.get_data[:p](x[1],x[2]+1), params), +, index_batch)
        
        function assert(data::Training_data)
            typeof(data) <: dataTarget{data_trajectory} &&  @warn "Target are not needed!"
            @assert !(typeof(data) <: data_sampled) "Need trajectories data!"
            @assert !(typeof(data) <: dataTarget{data_sampled}) "Need trajectories data!"
            
            @assert haskey(data.get_data, :q)
            @assert haskey(data.get_data, :p)

            nothing
        end

        new{typeof(sqdist),typeof(loss),typeof(assert)}(sqdist, loss, assert, min_length_trajectory)
    end
    
end

default_integrator(nn::LuxNeuralNetwork{<:SympNet}, data::data_trajectory) = BaseIntegrator()

# loss gradient
loss_gradient(nn::LuxNeuralNetwork{<:SympNet}, data, loss, index_batch, params = nn.params) = Zygote.gradient(p -> loss(nn, data, index_batch, p), params)[1]

# Loss function
#=
function loss(nn::LuxNeuralNetwork{<:SympNet}, q, p, params = nn.params, batch_size = 10)
    loss = 0
    #Index = sample(2:length(q), batch_size, replace = false)
    for i in 1:batch_size
        index = Int(ceil(rand()*(size(q,1)-1)))+1
        qp_new = nn(q[index-1,:], p[index-1,:], params)
        loss += norm(qp_new-[q[index,:]..., p[index,:]...])
    end
    loss
end

function full_loss(nn::LuxNeuralNetwork{<:SympNet}, q, p, params = nn.params)
    loss = 0
    #Index = 1:batch_size #sample(2:length(q), batch_size, replace = false)
    for index in 2:size(q,1)
        qp_new = nn(q[index-1,:], p[index-1,:], params)
        loss += norm(qp_new - [q[index,:]..., p[index,:]...])
    end
    loss
end

function grad_loss(nn::LuxNeuralNetwork{<:SympNet}, q, p, params = nn.params, batch_size = DEFAULT_BATCH_SIZE)
    Zygote.gradient(params -> loss(nn, q, p, params, batch_size), params)[1]
end


# Training

function train!(nn::LuxNeuralNetwork{<:SympNet}, m::AbstractMethodOptimiser, data_q, data_p; ntraining = DEFAULT_SYMPNET_NRUNS, batch_size = DEFAULT_BATCH_SIZE)
    
    #creation of optimiser
    opt = Optimizer(m,nn.model)

    # create array to store total loss
    total_loss = zeros(ntraining)

    # Learning runs
    @showprogress 1 "Training..." for j in 1:ntraining
        dp = grad_loss(nn, data_q, data_p, nn.params, batch_size)
        
        optimization_step!(opt, nn.model, nn.params, dp)
        total_loss[j] = full_loss(nn, data_q, data_p)
    end

    return total_loss
end
=#

function train!(nn::LuxNeuralNetwork{<:SympNet}, m::AbstractMethodOptimiser, data::Training_data; ntraining = DEFAULT_SYMPNET_NRUNS, hti::SympNetIntegrator = default_integrator(nn, data), batch_size_t = default_index_batch(data), showprogress::Bool = false)
    
    #verify that shape of data depending of the ExactIntegrator
    hti.assert(data)

    # create array to store total loss
    total_loss = zeros(ntraining)

    #creation of optimiser
    opt = Optimizer(m,nn.model)

    # Learning runs
    p = Progress(ntraining; enabled = showprogress)
    for j in 1:ntraining

        index_batch = get_batch(data, batch_size_t)

        dp = loss_gradient(nn, data, hti.loss, index_batch, nn.params) 

        #optimization_step!(opt, nn.model, nn.params, dp)

        total_loss[j] = hti.loss(nn, data)

        next!(p)
    end

    return total_loss
end

# Results

function Iterate_Sympnet(nn::LuxNeuralNetwork{<:SympNet}, q0, p0; n_points = DEFAULT_SIZE_RESULTS)

    n_dim = length(q0)
    
    # Array to store the predictions
    q_learned = zeros(n_points,n_dim)
    p_learned = zeros(n_points,n_dim)
    
    # Initialisation
    q_learned[1,:] = q0
    p_learned[1,:] = p0
    
    #Computation of phase space
    for i in 2:n_points
        qp_learned =  nn(q_learned[i-1,:], p_learned[i-1,:])
        q_learned[i,:] = qp_learned[1:n_dim]
        p_learned[i,:] = qp_learned[(1+n_dim):end]
    end

    return q_learned, p_learned
end