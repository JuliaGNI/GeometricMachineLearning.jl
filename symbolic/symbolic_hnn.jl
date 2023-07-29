include("symbolic_neural_network.jl")

struct SymbolicHNN{AT, ET, FT} <: SymbolicNeuralNetwork{AT, ET}
    nn::NeuralNetwork{AT}
    est::ET
    field::FT

    function SymbolicHNN(hnn::NeuralNetwork{<:HamiltonianNeuralNetwork})
        est, field = buildsymbolic(hnn)
        eval_est = eval(est)
        eval_field = eval(field)
        new{typeof(hnn.architecture), typeof(eval_est), typeof(eval_field )}(hnn, eval_est, eval_field )
    end

end

field(shnn::SymbolicHNN, x, params = params(shnn)) = shnn.field(x, develop(params)...)

Symbolize(hnn::NeuralNetwork{<:HamiltonianNeuralNetwork}) = SymbolicHNN(hnn)



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