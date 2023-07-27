include("symbolic_neural_network.jl")

struct SymbolicHNN{AT, ET, FT} <: SymbolicNeuralNetwork{AT, ET}
    nn::NeuralNetwork{AT}
    eval::ET
    field::FT


end

field(shnn::SymbolicHNN, x, params = params(snn)) = shnn.field(x, params)

