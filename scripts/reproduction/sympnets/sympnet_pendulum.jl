"""
Simple implementation of a pendulum for SympNets.
"""

using GeometricIntegrators
using GeometricMachineLearning
using LinearAlgebra
using CairoMakie
using Zygote
using ProgressMeter

# `smoke_size(full, smoke)` returns the second when GML_SMOKE is set, which is how the CI
# job runs this script end to end in seconds. See scripts/README.md.
include("../../utilities/smoke.jl")

# generate data for pendulum. `pendulum_data` returns `1 x n_time_steps` matrices, and everything
# below indexes one trajectory, so they are flattened to vectors here.
include("../../utilities/pendulum.jl")
const data = pendulum_data()
const q = vec(data.q)
const p = vec(data.p)
fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "q", ylabel = "p")
lines!(ax, q, p; label = "Training data.")

# Sympnet model.
#
# Two things moved out from under this script. `Gradient(n, upscaling, activation; change_q)` is
# gone -- the two cases are their own types, `GradientLayerQ` and `GradientLayerP` -- and the
# container is no longer Lux's: these layers are `AbstractNeuralNetworks` layers, which
# `Lux.Chain` rejects with "Encountered a non-AbstractLuxLayer in Chain." The `Chain` below is the
# one `GeometricMachineLearning` exports, and the network applies to `(input, parameters)` with no
# state to thread through.
model = Chain(GradientLayerQ(2, 10, tanh),
    GradientLayerP(2, 10, tanh),
    GradientLayerQ(2, 10, tanh),
    GradientLayerP(2, 10, tanh)
)

ps = NeuralNetwork(model, CPU(), Float64).params

# `index` names the point being predicted, so it starts at the second: the network maps the
# previous point onto it.
function loss_sing(ps, q, p, index)
    qp_new = model([q[index - 1], p[index - 1]], ps)
    norm(qp_new - [q[index], p[index]])
end

# defines a loss function
function loss(ps, q, p, batch_size = 10)
    loss = 0
    ntime = lastindex(q)
    for i in 1:batch_size
        index = rand(2:ntime)
        loss += loss_sing(ps, q, p, index)
    end
    loss
end

function full_loss(ps, q, p)
    loss = 0
    for i in 2:lastindex(q)
        loss += loss_sing(ps, q, p, i)
    end
    loss
end

# define momentum optimizer and initialize
method = AdamOptimizer()
# initial gradients for calling Cache constructor
opt = Optimizer(method, ps)
# `optimization_step!` takes the global section of the parameters, not the model.
λY = GlobalSection(ps)

# training 
println("initial loss: ", full_loss(ps, q, p))
training_steps = smoke_size(1000, 2)
@showprogress for i in 1:training_steps
    dp = Zygote.gradient(ps -> loss(ps, q, p), ps)[1]
    optimization_step!(opt, λY, ps, dp)
end
println("final loss: ", full_loss(ps, q, p))

# evaluate pendulum trajectory for the inital conditions for which it was trained
q_learned = zero(q)
p_learned = zero(p)
q_learned[1] = q[1]
p_learned[1] = p[1]

for i in 2:lastindex(q)
    q_learned[i], p_learned[i] = model([q_learned[i - 1], p_learned[i - 1]], ps)
end

# plot result and save figure to file
lines!(ax, q_learned, p_learned; label = "Learned trajectory.")
axislegend(ax)
CairoMakie.save("sympnet_pendulum.png", fig)
