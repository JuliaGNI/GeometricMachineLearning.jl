const DEFAULT_NRUNS = 1000

# The loss gradient function working for all types of arguments
loss_gradient(nn::AbstractNeuralNetwork, ti::AbstractTrainingIntegrator, data::AbstractTrainingData, index_batch, params = params(nn)) = Zygote.gradient(p -> loss(ti, nn, data, index_batch, p), params)[1]

####################################################################################
## Training on (LuxNeuralNetwork, AbstractTrainingData, OptimizerMethod, TrainingIntegrator, nruns, batch_size )
####################################################################################

"""
    train!(...)

Perform a training of a neural networks on data using given method a training integrator

Different ways of use:

    train!(neuralnetwork, data, optimizer = GradientOptimizer(1e-2), training_method; nruns = 1000, batch_size = default(data, type), showprogress = false )

# Arguments
- `neuralnetwork::LuxNeuralNetwork` : the neural net work using LuxBackend
- `data` : the data (see [`TrainingData`](@ref))
- `optimizer = GradientOptimizer`: the optimization method (see [`Optimizer`](@ref))
- `training_method` : specify the loss function used 
- `nruns` : number of iteration through the process with default value 
- `batch_size` : size of batch of data used for each step

"""
function train!(nn::AbstractNeuralNetwork, data_in::AbstractTrainingData, m::OptimizerMethod, ti::TrainingIntegrator{<:AbstractTrainingIntegrator} = default_integrator(nn, data); ntraining = DEFAULT_NRUNS, batch_size = missing, showprogress::Bool = false)

    # copy of data in the event of modification
    data = copy(data_in)

    # verify that dimension of data and input of nn match
    @assert dim(nn) == dim(data)

    # create an appropriate batch size by filling in missing values with default values
    bs = complete_batch_size(data, ti, batch_size)

    # check batch_size with respect to data
    check_batch_size(data, bs)

    # verify that shape of data depending of the ExactIntegrator
    data = matching(ti, data)

    # create array to store total loss
    total_loss = zeros(ntraining)

    #creation of optimiser
    opt = Optimizer(m, params(nn))

    # Learning runs
    p = Progress(ntraining; enabled = showprogress)
    for j in 1:ntraining
        index_batch = get_batch(data, bs; check = false)

        params_grad = loss_gradient(nn, ti, data, index_batch,  params(nn)) 

        optimization_step!(opt, model(nn), params(nn), params_grad)

        total_loss[j] = loss(ti, nn, data)

        next!(p)
    end

    return total_loss
end

####################################################################################
## Training on (LuxNeuralNetwork, AbstractTrainingData, TrainingParameters)
####################################################################################

"""
```julia
train!(neuralnetwork, data, optimizer, training_method; nruns = 1000, batch_size, showprogress = false )
```

# Arguments
- `neuralnetwork::LuxNeuralNetwork` : the neural net work using LuxBackend
- `data::AbstractTrainingData` : the data
- ``

"""
function train!(nn::NeuralNetwork{<:Architecture}, data::AbstractTrainingData, tp::TrainingParameters; showprogress::Bool = false)

    bs = complete_batch_size(data, method(tp), batchsize(tp))

    total_loss = train!(nn, data, opt(tp), method(tp); ntraining = nruns(tp), batch_size =  bs, showprogress = showprogress)

    sh = SingleHistory(tp, shape(data), size(data), total_loss)
    
    NeuralNetSolution(nn, sh, total_loss, problem(data), tstep(data))

end

####################################################################################
## Training on a TrainingSet structure
####################################################################################

train!(ts::TrainingSet; kwarsg...) = train!(nn(ts), data(ts), parameters(ts); kwarsg...)

train!(ts::TrainingSet...; kwarsg...) = train!(EnsembleTraining(ts...); kwarsg...)

####################################################################################
## Training on a EnsembleTraining structure
####################################################################################

function train!(ets::EnsembleTraining; kwarsg...)
    enns = EnsembleNeuralNetSolution()
    for ts in ets
        push!(enns,train!(ts::TrainingSet; kwarsg...))
    end
    enns
end

####################################################################################
## Training on a NeuralNetSolution with AbstractTrainingData and TrainingParameters
####################################################################################


function train!(nns::NeuralNetSolution, data::AbstractTrainingData, tp::TrainingParameters; kwarsg...)

    @assert tstep(data) == tstep(nns) || tstep(nns) == nothing || tstep(data) == nothing
    @assert problem(data) == problem(nns) || problem(nns) == nothing || problem(data) == nothing
    
    total_loss = train!(nn(nns), data, opt(tp), method(tp); ntraining = nruns(tp), batch_size = batchsize(tp))

    sh = SingleHistory(tp, shape(data), size(data), total_loss)

    update_history(nns, sh)
end

function train!(nns::NeuralNetSolution, ts::TrainingSet; kwarsg...)

    @assert nn(ts) == nn(nns)
    
    train!(nns::NeuralNetSolution, data(ts), parameters(ts); kwarsg...)

end