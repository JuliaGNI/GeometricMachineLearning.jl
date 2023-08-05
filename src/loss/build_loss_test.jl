using Symbolics
using GeometricMachineLearning
using AbstractNeuralNetworks
using SymbolicNeuralNetworks
import SymbolicNeuralNetworks: develop
using Zygote
using LinearAlgebra
using Distances

function get_string(expr)
    replace(replace(replace(string(expr), r"#= [^*\s]* =#" => ""), r"\n[\s]*\n" => "\n"), "Num" => "")
end

rdevelop(x) = x
rdevelop(t::Tuple{Any}) = [rdevelop(t[1])...]
rdevelop(t::Tuple) = [rdevelop(t[1])..., rdevelop(t[2:end])...]
function rdevelop(t::NamedTuple) 
   X = [[rdevelop(e)...] for e in t] 
   vcat(X...)
end


function get_track(W, SW, s)
    return W===SW ? (true, s) : (false, nothing)
end

function get_track(t::Tuple, W, s::String, info = 1)
    if length(t) == 1
        return get_track(t[1], W, string(s,"[",info,"]"))
    else
        bool, str = get_track(t[1], W, string(s,"[",info,"]"))
        return bool ? (true, str) : get_track(t[2:end], W, s, info+1)
    end
end

function get_track(nt::NamedTuple, W, s::String)
    for k in keys(nt)
        bool, str = get_track(nt[k], W, string(s,".",k))
        if bool
            return (bool, str)
        end
    end
    (false, nothing)
end

function rewrite(fun, SV, SX, ti, nn)
    for e in develop(SV)
        str_symbol = replace(string(e), r"\[.*"=>"")
        track = get_track(SV, e, "nt")[2]
        fun = Meta.parse(replace(string(fun), str_symbol => track))
    end
    for e in develop(SX)
        str_symbol = replace(string(e), r"\[.*"=>"")
        track = get_track(SX, e, "sargs")[2]
        fun = Meta.parse(replace(string(fun), str_symbol => track))
    end
    fun = Meta.parse(replace(string(fun), "SX" => "X"))
    fun = Meta.parse(replace(string(fun), r"function .*" => string("function ∇loss_single(::",typeof(ti),", ::",typeof(nn) ,", sargs, nt)\n")))
end


function build_hamiltonien(nn::NeuralNetwork{<:HamiltonianNeuralNetwork})

    # dimenstion of the input
    dimin = dim(nn.architecture)

    #compute the symplectic matrix
    sympmatrix = transposymplecticMatrix(dimin)
    
    # creates variables for the input
    @variables sinput[1:dimin]
    
    # creates variables for the parameters
    sparams = symbolic_params(nn)

    est = nn(sinput, sparams)

    field =  Symbolics.jacobian(est, sinput) * sympmatrix

    fun_est = build_function(est, sinput, develop(sparams)...)[2]
    fun_field = build_function(field, sinput, develop(sparams)...)[1]

    return (fun_est, fun_field)

end

function transposymplecticMatrix(n::Int) 
    I = Diagonal(ones(n÷2))
    Z = zeros(n÷2,n÷2)
    [Z -I; I Z]
end

nn= NeuralNetwork(HamiltonianNeuralNetwork(2, nhidden = 0), Float64)
_,v = build_hamiltonien(nn)
evalv = eval(v)
vectorfield(x, params) = evalv(x, develop(params)...)



function los(::TrainingIntegrator{SymplecticEulerA}, nn::AbstractNeuralNetwork{<:HamiltonianNeuralNetwork}, qₙ, qₙ₊₁, pₙ, pₙ₊₁, Δt, params = nn.params)
    dH = vectorfield([qₙ₊₁...,pₙ...], params)
    sqeuclidean(dH[1],(qₙ₊₁-qₙ)/Δt) + sqeuclidean(dH[2],(pₙ₊₁-pₙ)/Δt)
end


function build_gradloss(ti::TrainingIntegrator, nn::AbstractNeuralNetwork, args... ;params = nn.params)

    sargs = symbolic_params(args)[1]           
    sparams = symbolic_params(params)[1] 

    slos = los(ti, nn, sargs..., sparams)
    ∇slos = Symbolics.gradient(slos, rdevelop(sparams))

    sloss = build_function(slos, develop(sargs)..., develop(sparams)... )
    ∇loss = build_function(∇slos, develop(sargs)..., develop(sparams)... )[1]

    s = rewrite(∇loss, sparams, sargs, ti, nn)

    return (sloss, ∇loss, s)

end

method = SEuler()
A = [1,1,1,1,1]

fun, ∇fun, rew∇fun = build_gradloss(method,nn, A...)

r1 = eval(Meta.parse(get_string(∇fun)))(1,1,1,1,1,develop(nn.params)...)

r2 = eval(Meta.parse(get_string(rew∇fun)))(method, nn, A,nn.params)

r1==r2


#loss_path = Dict()
#loss_path[:c] = 1


function add_path(ti,nn)
    if (typeof(method), typeof(nn)) ∉ keys(loss_path)
        loss_path[(typeof(method), typeof(nn))] = loss_path[:c]
        loss_path[:c] += 1
    end
end

function access_path(ti, nn)
    return string(loss_path[(typeof(ti), typeof(nn))])
end

add_path(method, nn)

path ="/home/theoduez/Documents/Julia/GeometricMachineLearning.jl/src/loss/write_loss_"
write(path*access_path(method, nn)*".jl", get_string(rew∇fun))

#f = open("/home/theoduez/Documents/Julia/GeometricMachineLearning.jl/src/loss/write_loss.jl", "r")
