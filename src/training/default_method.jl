#=
    This file matches default methoddefault_method depending on the shape of data, its symbols, and the architecture of the neural network.
=#

function default_method(::AbstractBackend, ::AbstractTrainingData)
    throw(ArgumentError("Mismatch between the shape of data and the neural networks used to provide a default methoddefault_method for training"))
end

function default_method(::AbstractNeuralNetwork{<:HamiltonianArchitecture},
        ::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}, <:TrajectoryData})
    SEulerA()
end
function default_method(::AbstractNeuralNetwork{<:HamiltonianArchitecture},
        ::TrainingData{<:DataSymbol{<:DerivativePhaseSpaceSymbol}})
    ExactHnn()
end

function default_method(::AbstractNeuralNetwork{<:SympNet},
        ::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}, TrajectoryData})
    BasicSympNet()
end

function default_method(::AbstractNeuralNetwork{<:LagrangianNeuralNetwork},
        ::TrainingData{<:DataSymbol{<:PositionSymbol}, <:TrajectoryData})
    VariaMidPoint()
end
function default_method(::AbstractNeuralNetwork{<:LagrangianNeuralNetwork},
        ::TrainingData{<:DataSymbol{<:PosVeloAccSymbol}, <:SampledData})
    ExactLnn()
end
