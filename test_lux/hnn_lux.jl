using Lux
using Distances
using Zygote
using Optimisers
using ProgressMeter
using Random

# generate data
include("../scripts/data.jl")

data, target = get_data_set()
data = reshape(data,100,1)
target = reshape(target,100,1)

ld = 10
act = tanh

model = Chain(Dense(2,  ld, act),
              Dense(ld, ld, act),
              Dense(ld,  1, act))

# look at this in more detail; sets bias to zero for example
ps, st = Lux.setup(Random.default_rng(), model)

# define Hamiltonian via evaluation of network
function hnn(model, x, ps, st)
    y, state = Lux.apply(model, x, ps, st)
    return sum(y)
end

# compute Vector fields associated with network
grad_h(model,x,ps,st) = gradient(ξ -> hnn(model,ξ,ps,st), x)[1]
hnn_vf(model,x,ps,st) = [0 1; -1 0] * grad_h(model,x,ps,st)

# loss for a single datum
# loss_sing(model,x,y,ps,st) = sqeuclidean(hnn_vf(model,x,ps,st), y)
loss_sing(model,x,y,ps,st) = sqeuclidean(grad_h(model,x,ps,st), y)

# total loss
loss(model,x,y,ps,st) = mapreduce(i -> loss_sing(model,x[i],y[i],ps,st), +, eachindex(x,y))
# loss(model,x,y,ps,st) = mapreduce(z -> loss_sing(model, z..., ps, st), +, zip(x,y))

# loss gradient
# This doesn't work!!!!!
loss_gradient(model,x,y,ps,st) = gradient(Ω -> loss(model,x,y,Ω,st), ps)[1]


function train_flux_hnn(model,ps,st,data,target,runs)
        #create array to store total loss
        total_loss = zeros(runs)

        #do a couple learning runs
        @showprogress 1 "Training..." for j in 1:runs
                #gradient step
                batch = ceil.(Int, rand(10))
                step = loss_gradient(model, data[batch], target[batch], ps, st)

                #make gradient steps for all the model parameters W & b
                for Wb in ps
                        Wb .-= η .* step[Wb]
                end

                #total loss i.e. loss computed over all data
                total_loss[j] = loss(model, data, target, ps, st)
        end

        return (model, data, target, ps, st, total_loss)
end

train_flux_hnn(model, ps, st, data, target, 1)
