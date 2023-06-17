
const DEFAULT_LNN_NRUNS = 1000


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
(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork})(x, params = nn.params) = sum(apply(nn, x, params))

# gradient of the Lagrangian Neural Network
∇L(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, x, params = nn.params) = Zygote.gradient(x->nn(x, params), x)[1]

# hessian of the Lagrangian Neural Network
∇∇L(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, q, q̇, params = nn.params) = Zygote.hessian(x->nn(x, params),[q...,q̇...])

∇q̇∇q̇L(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, q, q̇, params = nn.params) = ∇∇L(nn, q, q̇, params)[(1+length(q̇)):end,(1+length(q̇)):end] 

∇q∇q̇L(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, q, q̇, params = nn.params) = ∇∇L(nn, q, q̇, params)[1:length(q),(1+length(q̇)):end] 

abstract type Lnn_training_integrator end


struct VariationalMidPointLNN{TD,TL,TA} <: Lnn_training_integrator
    sqdist::TD
    loss::TL
    assert::TA

    function VariationalMidPointLNN(;sqdist = sqeuclidean)

        function loss_single(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, qₙ₊₁, qₙ₊₂, Δt, params = nn.params)
            DL = ∇L(nn, Zygote.ignore([(qₙ₊₁+qₙ₊₂)/2..., (qₙ₊₁-qₙ)/Δt...]) , params)
            DL₁ = DL[1:length(qₙ)] 
            DL₂ = DL[length(qₙ)+1:end]
            sqdist(DL₁,-DL₂)
        end

        loss(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, datat::data_trajectory, index_batch = get_batch(datat), params = nn.params) =
        mapreduce(x->loss_single(nn, Zygote.ignore(datat.get_data[:q](x[1],x[2])), Zygote.ignore(datat.get_data[:q](x[1],x[2]+1)), Zygote.ignore(datat.get_data[:q](x[1],x[2]+2)), Zygote.ignore(datat.get_Δt()), params), +, index_batch)

        function assert(data::Training_data)
            typeof(data) <: dataTarget{data_trajectory} &&  @warn "Target are not needed!"
            @assert !(typeof(data) <: data_sampled) "Need trajectories data!"
            @assert !(typeof(data) <: dataTarget{data_sampled}) "Need trajectories data!"

            @assert haskey(data.get_data, :q)
      
            nothing
        end
        
        new{typeof(sqdist),typeof(loss),typeof(assert)}(sqdist, loss, assert)
    end
end



struct ExactIntegratorLNN{TD,TL,TA} <: Lnn_training_integrator
    sqdist::TD
    loss::TL
    assert::TA

    function ExactIntegratorLNN(;sqdist = sqeuclidean)

        function loss_single(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, q̇ₙ, q̈ₙ, params = nn.params)
            sum(∇q∇q̇L(nn,qₙ, q̇ₙ, params))  #inv(∇q̇∇q̇L(nn, qₙ, q̇ₙ, params))*(∇qL(nn, qₙ, q̇ₙ, params) - ∇q∇q̇L(nn, qₙ, q̇ₙ, params))
        end

        loss(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, datat::dataTarget{data_trajectory}, index_batch = get_batch(datat), params = nn.params) =
        mapreduce(x->loss_single(nn, datat.get_data[:q](x[1],x[2]), datat.get_data[:q̇](x[1],x[2]), datat.get_target[:q̈](x[1],x[2]), params), +, index_batch)

        loss(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, datat::dataTarget{data_sampled}, index_batch = get_batch(datat), params = nn.params) = 
        mapreduce(n->loss_single(nn, datat.get_data[:q](n), datat.get_data[:q̇](n), datat.get_target[:q̈](n), params), +, index_batch)

        function assert(data::Training_data)
            @assert (typeof(data) <: dataTarget) "Need targets for data!"

            @assert haskey(data.get_data, :q)
            @assert haskey(data.get_target, :q̇)
            @assert haskey(data.get_target, :q̈)
            
            nothing
        end

        new{typeof(sqdist),typeof(loss),typeof(assert)}(sqdist, loss, assert)
    end
end

default_integrator(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, data::dataTarget) = ExactIntegratorLNN()
default_integrator(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, data::data_trajectory) = VariationalMidPointLNN()

# loss gradient
loss_gradient(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, data, loss, index_batch, params = nn.params) = Zygote.gradient(p -> loss(nn, data, index_batch,p), params)[1]


# training
function train!(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, m::AbstractMethodOptimiser, data::Training_data; ntraining = DEFAULT_HNN_NRUNS, lti::Lnn_training_integrator = default_integrator(nn, data), batch_size_t = default_index_batch(data), showprogress::Bool = false)
    
    #verify that shape of data depending of the ExactIntegrator
    lti.assert(data)
    
    # create array to store total loss
    total_loss = zeros(ntraining)

    #creation of optimiser
    opt = Optimizer(m,nn.model)

    # convert parameters to tuple
    params_tuple = Tuple([Tuple(x) for x in nn.params])

    keys_1 = keys(nn.params)
    keys_2 = [keys(x) for x in values(nn.params)]

    # Learning runs
    p = Progress(ntraining; enabled = showprogress)
    for j in 1:ntraining

        index_batch = get_batch(data, batch_size_t)
    
        params_grad = loss_gradient(nn, data, lti.loss, index_batch, params_tuple) 
    
        dp = NamedTuple(zip(keys_1,[NamedTuple(zip(k,x)) for (k,x) in zip(keys_2,params_grad)]))
    
        optimization_step!(opt, nn.model, nn.params, dp)
    
        total_loss[j] = lti.loss(nn, data)

        next!(p)
    end

    return total_loss
end
