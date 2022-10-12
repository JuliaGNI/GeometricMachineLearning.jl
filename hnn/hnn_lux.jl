using Distances
using Lux
using Optimisers
using ProgressMeter
using Random
using Zygote


# define some custom apply methods for Chain and Dense
# that use Tuples for parameters instead of NamedTuples
# and do not return a state but only the results of each
# layer and the whole chain
# splitting of Lux's return tuple of (result, state) as well
# as symbolic indexing of NamedTuples does not work when
# computing two derivatives with Zygote

@generated function Lux.applychain(layers::NamedTuple{fields}, x, ps::Tuple, st::NamedTuple{fields}) where {fields}
    N = length(fields)
    x_symbols = vcat([:x], [gensym() for _ in 1:N])
    calls = [:(($(x_symbols[i + 1])) = Lux.apply(layers.$(fields[i]),
                                                $(x_symbols[i]),
                                                ps[$i],
                                                st.$(fields[i]))) for i in 1:N]
    push!(calls, :(return $(x_symbols[N + 1])))
    return Expr(:block, calls...)
end

@inline function Lux.apply(d::Dense{false}, x::AbstractVecOrMat, ps::Tuple, st::NamedTuple)
    return d.activation.(ps[1] * x)
end

@inline function Lux.apply(d::Dense{true}, x::AbstractVector, ps::Tuple, st::NamedTuple)
    return d.activation.(ps[1] * x .+ vec(ps[2]))
end


# generate data
include("../scripts/data.jl")

data, target = get_data_set()
data = reshape(data,100,1)
target = reshape(target,100,1)

ld = 10
act = tanh

model = Chain(Dense(2,  ld, act),
              Dense(ld, ld, act),
              Dense(ld,  1; bias=false))

# look at this in more detail; sets bias to zero for example
ps, st = Lux.setup(Random.default_rng(), model)

# define Hamiltonian via evaluation of network
function hnn(model, x, params::Tuple, state)
    y = Lux.apply(model, x, params, state)
    return sum(y)
end

function hnn(model, x, params::NamedTuple, state)
    y, st = Lux.apply(model, x, params, state)
    return sum(y)
end

# compute vector fields associated with network
grad_h(model, x, params, state) = Zygote.gradient(ξ -> hnn(model, ξ, params, state), x)[1]
hnn_vf(model, x, params, state) = [0 1; -1 0] * grad_h(model, x, params, state)

# loss for a single datum
loss_sing(model, x, y, params, state) = sqeuclidean(grad_h(model, x, params, state), y)

# total loss
hnn_loss(model, x, y, params, state) = mapreduce(i -> loss_sing(model, x[i], y[i], params, state), +, eachindex(x,y))

# loss gradient
hnn_loss_gradient(model, loss, x, y, params, state) = Zygote.gradient(p -> loss(model, x, y, p, state), params)[1]


function train_lux_hnn(model, loss, params, state, data, target, runs, η=0.001)
    # create array to store total loss
    total_loss = zeros(runs)

    # convert parameters to tuple
    params_tuple = Tuple([Tuple(x) for x in params])

    # do a couple learning runs
    @showprogress 1 "Training..." for j in 1:runs
        # gradient step
        batch = ceil.(Int, rand(10))
        params_grad = hnn_loss_gradient(model, loss, data[batch], target[batch], params_tuple, state)

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

train_lux_hnn(model, hnn_loss, ps, st, data, target, 1)
