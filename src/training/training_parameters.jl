#= 
    TrainingParameters brings together all the parameters required for training, which are :
        - nruns: number of iterations for training,
        - method: the training integrator used in the loss function,
        - mopt: the method of optimization,
        - bs: the size of the subset of data used at each stage and chosen at random to calculate the loss.
=#

struct TrainingParameters{TM, TO, Tbatch}
    nruns::Int
    method::TM
    mopt::TO
    bs::Tbatch

    # `mopt` used to default to `default_optimizer()`, which was deleted with the rest of GML's
    # optimizer layer when it moved to GeometricOptimizers — the default was left behind and every
    # call that took it raised `UndefVarError`. The optimizer is now given explicitly, as the
    # changelog already said it was.
    function TrainingParameters(nruns, method, mopt; batch_size = missing)
        new{typeof(method), typeof(mopt), typeof(batch_size)}(nruns, method, mopt, batch_size)
    end
end

function TrainingParameters(tp::TrainingParameters; nruns = nruns(tp), method = method(tp), opt = opt(tp), batch_size = batchsize(tp))
    TrainingParameters(nruns, method, opt; batch_size = batch_size)
end

# This took neither an optimizer nor a training method: it called `default_optimizer()` and
# `default_integrator(nn, data)`, and *neither* exists — the second has been `default_method` for
# some time. Both are supplied explicitly now.
function TrainingParameters(nn::LuxNeuralNetwork, data::AbstractTrainingData, mopt;
        method = default_method(nn, data), nruns = DEFAULT_NRUNS)
    batch_size = complete_batch_size(data, method, missing)
    TrainingParameters(nruns, method, mopt; batch_size = batch_size)
end

@inline nruns(tp::TrainingParameters) = tp.nruns
@inline method(tp::TrainingParameters) = tp.method
@inline opt(tp::TrainingParameters) = tp.mopt
@inline batchsize(tp::TrainingParameters) = tp.bs

