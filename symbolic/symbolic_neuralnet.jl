include("abstract_symbolic_neuralnet.jl")

struct SymbolicNeuralNetwork{AT, ET} <: AbstractSymbolicNeuralNetwork{AT, ET}
    nn::NeuralNetwork{AT}
    est::ET

    function SymbolicNeuralNetwork(nn::NeuralNetwork)
        est = buildsymbolic(nn)
        eval_est = eval(est)
        new{typeof(nn.architecture), typeof(eval_est)}(nn, eval_est)
    end

end


Symbolize(nn::NeuralNetwork) = SymbolicNeuralNetwork(nn)


function buildsymbolic(nn::NeuralNetwork)

    @variables sinput[1:dim(nn.architecture)]
    
    sparams = symbolicParams(nn)

    est = nn(sinput, sparams)

    build_function(est, sinput, develop(sparams)...)[2]

end