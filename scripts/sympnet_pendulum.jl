"""
Simple implementation of a pendulum for SympNets.
"""

using GeometricIntegrators
using GeometricMachineLearning
using LinearAlgebra
using Lux
using Plots
using Zygote

import Random 


# generate data for pendulum
include("pendulum.jl")
q, p = pendulum_data()
plt = plot(q, p, label="Training data.")

# Sympnet model 
model = Chain(  Gradient(2, 10, tanh), 
                Gradient(2, 10, tanh; change_q=false)
                #Gradient(2, 10, tanh),
                #Gradient(2, 10, tanh; change_q=false)
                )

ps, st = Lux.setup(Random.default_rng(), model)

# defines a loss function 
function loss(ps, q, p, batch_size=10)
    loss = 0 
    ntime = lastindex(q)
    for i in 1:batch_size
        index = Int(ceil(rand()*ntime))
        qp_new = Lux.apply(model, [q[index-1], p[index-1]], ps, st)[1]
        loss += norm(qp_new - [q[index], p[index]])
    end
    loss 
end

function full_loss(ps, q, p)
    loss = 0 
    for i in 1:lastindex(q)
        qp_new = Lux.apply(model, [q[i-1], p[i-1]], ps, st)[1]
        loss += norm(qp_new - [q[i], p[i]])
    end
    loss 
end 

# define momentum optimizer and initialize
o = AdamOptimizer()
# initial gradients for calling Cache constructor
cache = init_optimizer_cache(model, o)

# training 
println("initial loss: ", full_loss(ps, q, p))
training_steps = 1000
for i in 1:training_steps
    dp = Zygote.gradient(ps -> loss(ps, q, p), ps)[1]
    optimization_step!(o, model, ps, cache, dp)
end 
println("final loss: ", full_loss(ps, q, p))

# evaluate pendulum trajectory for the inital conditions for which it was trained
q_learned = zero(q)
p_learned = zero(p)
q_learned[0] = q[0]
p_learned[0] = p[0]

for i in 1:lastindex(q)
    q_learned[i], p_learned[i] = Lux.apply(model, [q_learned[i-1], p_learned[i-1]], ps, st)[1]
end 

# plot result and save figure to file
plot(plt, q_learned, p_learned, label="Learned trajectory.")
savefig("sympnet_pendulum.png")
