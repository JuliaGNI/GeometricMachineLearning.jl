abstract type  AbstractTrainingIntegrator end

abstract type HnnTrainingIntegrator <: AbstractTrainingIntegrator end
abstract type LnnTrainingIntegrator <: AbstractTrainingIntegrator end
abstract type SympNetTrainingIntegrator <: AbstractTrainingIntegrator end

function loss end
function loss_single end

# Structure 

#Define common strucutre integrator
struct TrainingIntegrator{TIT,TD}
    type::TIT
    sqdist::TD
end

type(ti::TrainingIntegrator) = ti.type


# Assertion for good usage of training integrator

required_key(ti::AbstractTrainingIntegrator) = @warn "No recquired_key functions for "*string(typeof(ti))*"!"

data_goal(ti::AbstractTrainingIntegrator) = @warn "No data recquirement for "*string(typeof(ti))*". Errors may occur."; nothing

function assert(ti::AbstractTrainingIntegrator, data::AbstractTrainingData)
    #assert(data_goal(ti), data)

    for key in required_key(ti)
        @assert (haskey(data.get_data, key) || haskey(data.get_target, key)) "You forgot the key "*string(key)*"!"
    end

end





#loss gradient
loss_gradient(nn::LuxNeuralNetwork{<:AbstractArchitecture}, ti::AbstractTrainingIntegrator, data, index_batch, params = nn.params) = Zygote.gradient(p -> loss(ti, nn, data, index_batch, p), params)[1]

const DEFAULT_NRUNS = 1000

function train!(nn::LuxNeuralNetwork{<:AbstractArchitecture}, m::AbstractMethodOptimiser, data::AbstractTrainingData; ntraining = DEFAULT_NRUNS, ti::AbstractTrainingIntegrator = default_integrator(nn, data), batch_size_t = default_index_batch(data), showprogress::Bool = false)
    
    #verify that shape of data depending of the ExactIntegrator
    assert(ti, data)

    # create array to store total loss
    total_loss = zeros(ntraining)

    #creation of optimiser
    opt = Optimizer(m, nn.model)

    # transform parameters (if needed) to match with Zygote
    params_tuple, keys =  pretransform(ti, nn.params)

    # Learning runs
    p = Progress(ntraining; enabled = showprogress)
    for j in 1:ntraining
        index_batch = get_batch(data, batch_size_t)

        params_grad = loss_gradient(nn, ti, data, index_batch, params_tuple) 

        dp = posttransform(ti, params_grad, keys)

        optimization_step!(opt, nn.model, nn.params, dp)

        total_loss[j] = loss(ti, nn, data)

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



