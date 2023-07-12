
const DEFAULT_NRUNS = 1000

#loss gradient
loss_gradient(nn::LuxNeuralNetwork{<:AbstractArchitecture}, ti::AbstractTrainingIntegrator, data, index_batch, params = nn.params) = Zygote.gradient(p -> loss(ti, nn, data, index_batch, p), params)[1]


#train function
function train!(nn::LuxNeuralNetwork{<:AbstractArchitecture}, m::OptimizerMethod, data::AbstractTrainingData; ntraining = DEFAULT_NRUNS, ti::TrainingIntegrator{<:AbstractTrainingIntegrator} = default_integrator(nn, data), batch_size_t = default_index_batch(data,type(ti)), showprogress::Bool = false)
    
    #verify that shape of data depending of the ExactIntegrator
    assert(type(ti), data)

    # create array to store total loss
    total_loss = zeros(ntraining)

    #creation of optimiser
    opt = Optimizer(m, nn.model)

    # transform parameters (if needed) to match with Zygote
    params_tuple, keys =  pretransform(type(ti), nn.params)

    # Learning runs
    p = Progress(ntraining; enabled = showprogress)
    for j in 1:ntraining
        index_batch = get_batch(data, batch_size_t)

        params_grad = loss_gradient(nn, type(ti), data, index_batch, params_tuple) 

        dp = posttransform(type(ti), params_grad, keys)

        optimization_step!(opt, nn.model, nn.params, dp)

        total_loss[j] = loss(type(ti), nn, data)

        next!(p)
    end

    return total_loss
end


pretransform(::AbstractTrainingIntegrator, params::NamedTuple) = params, nothing

posttransform(::AbstractTrainingIntegrator, params,  args...) = params


TuppleNeededTrainingIntegrator = Union{HnnTrainingIntegrator, LnnTrainingIntegrator}

function pretransform(::TuppleNeededTrainingIntegrator, params::NamedTuple)

    # convert parameters to tuple
    params_tuple = Tuple([Tuple(x) for x in params])

    keys_1 = keys(params)
    keys_2 = [keys(x) for x in values(params)]

    params_tuple, (keys_1, keys_2)
end


function posttransform(::HnnTrainingIntegrator, params_grad::Tuple, keys)

    NamedTuple(zip(keys[1],[NamedTuple(zip(k,x)) for (k,x) in zip(keys[2],params_grad)]))

end



