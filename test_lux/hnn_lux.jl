using Lux
using Distances
using Zygote
using Optimisers
using ProgressMeter
using Random

#generate data
include("../scripts/data.jl")

data,target = get_data_set()
data = reshape(data,100,1)
target = reshape(target,100,1)

ld = 10

model = Chain(Dense(2, ld, tanh),
                Dense(ld,   ld, tanh),
                Dense(ld,    1, tanh))

#look at this in more detail; sets bias to zero for example
ps,st = Lux.setup(Random.default_rng(),model)

#compute Vector fields associated with network

hnn_vf(model,x,ps,st) = [0 1; -1 0] * gradient(ξ -> sum(Lux.apply(model,ξ,ps,st)[1]),x)[1]
#loss for a single datum
loss_sing(model,x,y,ps,st) = sqeuclidean(hnn_vf(model,x,ps,st),y)
loss(model,x,y,ps,st) = mapreduce(i -> loss_sing(model,x[i],y[i],ps,st),+,eachindex(x,y))


#This doesn't work!!!!!
loss_gradient(model,x,y,ps,st) = gradient(Ω -> loss(model,x,y,Ω,st),ps)[1]


function train_flux_hnn(model,ps,st,data,target,runs)
        #create array to store total loss
        total_loss = zeros(runs)

        #do a couple learning runs
        @showprogress 1 "Training..." for j in 1:runs
                #gradient step
                batch = ceil.(Int,rand(10))
                step = loss_gradient(model,data[batch],target[batch],ps,st)

                #make gradient steps for all the model parameters W & b
                for Wb in ps
                        Wb .-= η .* step[Wb]
                end

                #total loss i.e. loss computed over all data
                total_loss[j] = loss(model,data,target,ps,st)
        end

        return (model, data, target, ps, st, total_loss)
end

train_flux_hnn(model,ps,st,data,target,1)
