using GeometricMachineLearning
using LinearAlgebra
using Symbolics

using AbstractNeuralNetworks
import AbstractNeuralNetworks: NeuralNetwork

include("utils.jl")
include("symbolic_params.jl")


function symbolicModel(nn::NeuralNetwork)
    @variables sinput[1:dim(nn.architecture)]
    sparams =  symbolicParams(nn)
    nn(sinput, sparams)
end


function symbolicField(nn::NeuralNetwork{<:HamiltonianNeuralNetwork})


end


function buildsymbolic(nn::NeuralNetwork{<:HamiltonianNeuralNetwork})

    # dimenstion of the input
    dimin = dim(nn.architecture)

    #compute the symplectic matrix
    sympmatrix = symplecticMatrix(dimin)
    
    # creates variables for the input
    @variables sinput[1:dimin]
    
    # creates variables for the parameters
    sparams = symbolicParams(nn)

    est = nn(sinput, sparams)
    
    Dᵢₙₚᵤₜ = Differential(sinput)
    
    field =  sympmatrix * Dᵢₙₚᵤₜ(nn(sinput, sparams))
    
    fun_est = build_function(est, sinput, develop(sparams)...)[2]
    fun_field = build_function(field, sinput, develop(sparams)...)[2]

    return (fun_est, fun_field)

end


hnn = HamiltonianNeuralNetwork(2)
nn = NeuralNetwork(hnn, Float64)

symbolicParams(nn)

(fun_est, fun_field) =  buildsymbolic(nn)




write("symbolic/est.jl",   get_string(fun_est))
write("symbolic/field.jl", get_string(fun_field))







