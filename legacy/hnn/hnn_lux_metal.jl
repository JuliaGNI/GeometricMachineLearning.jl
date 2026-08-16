using Distances
using LinearAlgebra
using Lux
using Metal
using Optimisers
using ProgressMeter
using Random
using Zygote

using GeometricMachineLearning: get_batch


# generate data
include("../../scripts/data.jl")

data, target = get_data_set()
data = [MtlArray(Float32.(d)) for d in data]
target = [MtlArray(Float32.(t)) for t in target]


# learning rate
const η = Float32(.001)

# number of training runs
const nruns = Int32(1000)

# layer width
const ld = 5

# activation function
const act = tanh

# create model
model = Chain(Dense(2,  ld, act),
              Dense(ld, ld, act),
              Dense(ld,  1; bias=false))

# look at this in more detail; sets bias to zero for example
ps, st = Lux.setup(Random.default_rng(), model)

ps = NamedTuple{keys(ps)}([NamedTuple{keys(x)}([MtlArray(Float32.(a)) for a in x]) for x in ps])


# define Hamiltonian via evaluation of network
function hnn(model, x, params::Tuple, state)
    y = Lux.apply(model, x, params, state)
    dot(MtlVector(ones(Float32,1)), y)
end

function hnn(model, x, params::NamedTuple, state)
    y, st = Lux.apply(model, x, params, state)
    dot(MtlVector(ones(Float32,1)), y)
end

# compute vector fields associated with network
grad_h(model, x, params, state) = Zygote.gradient(ξ -> hnn(model, ξ, params, state), x)[1]
# hnn_vf(model, x, params, state) = [0 1; -1 0] * grad_h(model, x, params, state)
hnn_vf(model, x, params, state) = grad_h(model, x, params, state)

# loss for a single datum
loss_sing(model, x, y, params, state) = sqeuclidean(hnn_vf(model, x, params, state), y)

# total loss
hnn_loss(model, x, y, params, state) = mapreduce(i -> loss_sing(model, x[i], y[i], params, state), +, eachindex(x,y))

# loss gradient
hnn_loss_gradient(model, loss, x, y, params, state) = Zygote.gradient(p -> loss(model, x, y, p, state), params)[1]


function train_lux_hnn(model, loss, params, state, data, target, runs, η)
    # create array to store total loss
    total_loss = MtlArray(zeros(Float32, runs))

    # convert parameters to tuple
    params_tuple = Tuple([Tuple(x) for x in params])

    # do a couple learning runs
    @showprogress 1 "Training..." for j in 1:runs
        # gradient step
        params_grad = hnn_loss_gradient(model, loss, get_batch(data, target)..., params_tuple, state)

        # make gradient steps for all the model parameters W & b
        for i in eachindex(params_tuple, params_grad)
            for (p, dp) in zip(params_tuple[i], params_grad[i])
                p .-= η .* dp
            end
        end

        # total loss i.e. loss computed over all data
        total_loss[j] = loss(model, data, target, params, state)
    end

    return (model, data, target, params, state, total_loss)
end

model, data, target, params, state, total_loss = train_lux_hnn(model, hnn_loss, ps, st, data, target, nruns, η)

#time training (after warmup)
# train_lux_hnn(model, hnn_loss, ps, st, data, target, nruns, η)
# @time model, data, target, params, state, total_loss = train_lux_hnn(model, hnn_loss, ps, st, data, target, nruns, η)

# profile training
# run with julia --track-allocation=user hnn.jl
# Profile.clear()
# Profile.clear_malloc_data()
# @profile model, data, target, params, state, total_loss = train_lux_hnn(model, hnn_loss, ps, st, data, target, nruns, η)

# learned Hamiltonian & vector field
hnn_est(ξ) = hnn(model, ξ, params, state)
dhnn_est(ξ) = hnn_vf(model, ξ, params, state)

# plot results

include("../../scripts/plots.jl")

plot_hnn(H, hnn_est, total_loss; filename="hnn_lux.png")
