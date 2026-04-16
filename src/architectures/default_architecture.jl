
#=
    This file gives a default architecture based on the data provided. It is useful for Ense;bleSolution.
=#

default_arch(::AbstractTrainingData, ::Int) = throw(ArgumentError("It is not possible to establish a default architecture based on the data provided."))
default_arch(::TrainingData{<:PositionSymbol}, ninput::Int) = LagrangianNeuralNetwork(ninput)
default_arch(::TrainingData{<:PhaseSpaceSymbol, TrajectoryData}, ninput::Int) = HamiltonianArchitecture(ninput)
default_arch(::TrainingData{<:DerivativePhaseSpaceSymbol}, ninput::Int) = GSympNet(ninput)

