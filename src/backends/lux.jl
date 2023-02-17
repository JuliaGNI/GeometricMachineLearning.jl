
import Lux

struct LuxBackend end

struct LuxNeuralNetwork{AT,MT,PT,ST} <: AbstractNeuralNetwork
    architecture::AT
    model::MT
    params::PT
    state::ST
end

function NeuralNetwork(arch::AbstractArchitecture, back::LuxBackend)
    # create model
    model = chain(arch, back::LuxBackend)

    # initialize Lux params and state
    params, state = Lux.setup(Random.default_rng(), model)

    # create Lux neural network
    LuxNeuralNetwork(arch, model, params, state)
end

function(nn::LuxNeuralNetwork)(x)
    # apply network
    y, st = Lux.apply(nn.model, x, nn.params, nn.state)
    
    # update state
    nn.state .= st

    # sum output to obtain (scalar) result
    return sum(y)
end


function update_layer!(::Lux.AbstractExplicitLayer, x, dx, η)
    for obj in keys(x)
        x[obj] .-= η * dx[obj]
    end
end


# define some custom apply methods for Chain and Dense
# that use Tuples for parameters instead of NamedTuples
# and do not return a state but only the results of each
# layer and the whole chain
# splitting of Lux's return tuple of (result, state) as well
# as symbolic indexing of NamedTuples does not work when
# computing two derivatives with Zygote as is required for
# Hamiltonian Neural Networks

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

@inline function Lux.apply(d::Lux.Dense{false}, x::AbstractVecOrMat, ps::Tuple, st::NamedTuple)
    return d.activation.(ps[1] * x)
end

@inline function Lux.apply(d::Lux.Dense{true}, x::AbstractVector, ps::Tuple, st::NamedTuple)
    return d.activation.(ps[1] * x .+ vec(ps[2]))
end
