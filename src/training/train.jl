const DEFAULT_NRUNS = 1000

# The loss gradient function working for all types of arguments
loss_gradient(nn::NeuralNetwork{<:Architecture}, ti::AbstractTrainingMethod, data::AbstractTrainingData, index_batch, params = nn.params) = Zygote.gradient(p -> loss(ti, nn, data, index_batch, p), params)[1]

loss_gradient(loss, index_batch, params = nn.params) = Zygote.gradient(p -> loss(p, index_batch, params)[1]

#loss_gradient(nn::SymbolicNeuralNetwork, ti::AbstractTrainingMethod, data::AbstractTrainingData, index_batch, params = params(nn)) = 
#mapreduce(args->∇loss_single(ti, nn, get_loss(ti, nn, data, args)..., params), +, index_batch)


####################################################################################
## Training on (LuxNeuralNetwork, AbstractTrainingData, OptimizerMethod, TrainingMethod, nruns, batch_size )
####################################################################################

"""
    train!(...)

Perform a training of a neural networks on data using given method a training Method

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
function train!(nn::AbstractNeuralNetwork, _data::AbstractTrainingData, m::OptimizerMethod, method = default_method(nn, data); ntraining = DEFAULT_NRUNS, batch_size = missing, showprogress::Bool = false, timer::Bool = false)

    # create a timer
    to = TimerOutput()

    # copy of data in the event of modification
    data = copy(_data)

    # verify that dimension of data and input of nn match
    @assert dim(nn) == dim(data)

    # create an appropriate batch size by filling in missing values with default values
    typeof(method) <: TrainingMethod ? (@timeit to "Complete BatchSize" bs = complete_batch_size(data, method, batch_size)) : nothing

    # check batch_size with respect to data
    @timeit to "Check BatchSize" check_batch_size(data, bs)

    # verify that shape of data depending of the Integrator
    typeof(method) <: TrainingMethod ? data = matching(method, data) : nothing

    # creation of the loss function
    loss(params, batch) =  typeof(method) <: TrainingMethod ? loss(method, nn, data, batch, params) : method(nn, data, batch, params)

    # creation of optimiser
    opt = Optimizer(m, params(nn))

    # creation of the array to store total loss
    total_loss = zeros(ntraining)

    # Learning runs
    p = Progress(ntraining; enabled = showprogress)
    for j in 1:ntraining
        index_batch = get_batch(data, bs; check = false)

        ∇params = loss_gradient(loss, index_batch,  params(nn)) 

        optimization_step!(opt, model(nn), params(nn), ∇params)

        total_loss[j] = loss(ti, nn, data)

        next!(p)
    end

    timer ? show(to) : nothing

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
function train!(nn::AbstractNeuralNetwork{<:Architecture}, data::AbstractTrainingData, tp::TrainingParameters; kwarsg...)

    bs = complete_batch_size(data, method(tp), batchsize(tp))

    total_loss = train!(nn, data, opt(tp), method(tp); ntraining = nruns(tp), batch_size =  bs, kwarsg...)

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


function train!(nns::NeuralNetSolution, data::AbstractTrainingData, tp::TrainingParameters; kwargs...)

    @assert tstep(data) == tstep(nns) || tstep(nns) == nothing || tstep(data) == nothing
    @assert problem(data) == problem(nns) || problem(nns) == nothing || problem(data) == nothing
    
    total_loss = train!(nn(nns), data, opt(tp), method(tp); ntraining = nruns(tp), batch_size = batchsize(tp), kwargs...)

    sh = SingleHistory(tp, shape(data), size(data), total_loss)

    update_history(nns, sh)
end

function train!(nns::NeuralNetSolution, ts::TrainingSet; kwarsg...)

    @assert nn(ts) == nn(nns)
    
    train!(nns::NeuralNetSolution, data(ts), parameters(ts); kwarsg...)

end