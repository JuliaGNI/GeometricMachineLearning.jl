"""
Simple implementation of a pendulum for SympNets.
"""

using GeometricIntegrators
using GeometricMachineLearning
using LinearAlgebra
using Lux
using Plots
using Zygote
using ProgressMeter
using CUDA

import Random 


# generate data for pendulum
include("pendulum.jl")
q, p = pendulum_data()
plt = plot(q, p, label="Training data.")
q = q |> cu 
p = p |> cu

# Sympnet model 
model = Chain(  Gradient(2, 10, tanh),
                Gradient(2, 10, tanh; change_q=false),
                Gradient(2, 10, tanh),
                Gradient(2, 10, tanh; change_q=false)
                )

ps, st = Lux.setup(CUDA.device(), Random.default_rng(), model)

function get_z_from_qp!(z, q, p, index)
    i = threadIdx().x 
    z[i] = q[index]
    z[i + 1] = p[index]
    return
end

function loss_sing(ps, q, p, index) 
    z = CUDA.zeros(2)
    z_applied = CUDA.zeros(2)
    @cuda threads=length(q) get_z_from_qp!(z, q, p, index-1)
    @cuda threads=length(q) get_z_from_qp!(z_applied, q, p, index)
    qp_new = Lux.apply(model, z, ps, st)[1]
    norm(qp_new - z_applied)
end

# defines a loss function 
function loss(ps, q, p, batch_size=10)
    loss = 0 
    ntime = lastindex(q)
    for i in 1:batch_size
        index = Int(ceil(rand()*ntime))
        loss += loss_sing(ps, q, p, index)
    end
    loss 
end

function full_loss(ps, q, p)
    loss = 0 
    for i in 1:lastindex(q)
        loss += loss_sing(ps, q, p, i)
    end
    loss 
end 

# define momentum optimizer and initialize
method = AdamOptimizer()
# initial gradients for calling Cache constructor
opt = Optimizer(method, model)

# training 
println("initial loss: ", full_loss(ps, q, p))
training_steps = 1000
@showprogress for i in 1:training_steps
    dp = Zygote.gradient(ps -> loss(ps, q, p), ps)[1]
    optimization_step!(opt, model, ps, dp)
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
