
#=
    This file gives a default architecture based on the data provided. It is useful for Ense;bleSolution.
=#

function default_arch(::AbstractTrainingData, ::Int)
    throw(ArgumentError("It is not possible to establish a default architecture based on the data provided."))
end
function default_arch(::TrainingData{<:PositionSymbol}, ninput::Int)
    LagrangianNeuralNetwork(ninput)
end
function default_arch(::TrainingData{<:PhaseSpaceSymbol, TrajectoryData}, ninput::Int)
    HamiltonianArchitecture(ninput)
end
default_arch(::TrainingData{<:DerivativePhaseSpaceSymbol}, ninput::Int) = GSympNet(ninput)
