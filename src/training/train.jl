
const DEFAULT_NRUNS = 1000

#loss gradient
loss_gradient(nn::LuxNeuralNetwork{<:AbstractArchitecture}, ti::AbstractTrainingIntegrator, data, index_batch, params = nn.params) = Zygote.gradient(p -> loss(ti, nn, data, index_batch, p), params)[1]


#train function
function train!(nn::LuxNeuralNetwork{<:AbstractArchitecture}, m::AbstractMethodOptimiser, data::AbstractTrainingData; ntraining = DEFAULT_NRUNS, ti::TrainingIntegrator{<:AbstractTrainingIntegrator} = default_integrator(nn, data), batch_size_t = default_index_batch(data,type(ti)), showprogress::Bool = false)
    
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


function train!(nn::LuxNeuralNetwork{<:AbstractArchitecture}, data::AbstractTrainingData, tp::TrainingParameters; showprogress::Bool = false)

    total_loss = train!(nn, opt(tp), data; ntraining = nruns(tp), ti = method(tp), batch_size_t = batch_size(tp), showprogress = showprogress)

    sh = SingleHistory(tp, shape(data), size(data), total_loss)
    
    NeuralNetSolution(nn, total_loss, sh, problem(data), tstep(data))

end


function train!(nns::NeuralNetSolution, data::AbstractTrainingData, tp::TrainingParameters; showprogress::Bool = false)

    @assert tstep(data) == tstep(nns) || tstep(nns) == nothing || tstep(data) == nothing
    @assert problem(data) == problem(nns) || problem(nns) == nothing || problem(data) == nothing
    
    total_loss = train!(nn(nns), opt(tp), data; ntraining = nruns(tp), ti = method(tp), batch_size_t = batch_size(tp), showprogress = showprogress)

    sh = SingleHistory(tp, shape(data), size(data), total_loss)

    new_history = _update(nns(history), sh)
    
    NeuralNetSolution(nn(nns), total_loss, new_history, problem(nns), tstep(nns))

end




