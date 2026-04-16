#=
    This file matches default methoddefault_method depending on the shape of data, its symbols, and the architecture of the neural network.
=#

default_method(::AbstractBackend, ::AbstractTrainingData) = throw(ArgumentError("Mismatch between the shape of data and the neural networks used to provide a default methoddefault_method for training"))

default_method(::AbstractNeuralNetwork{<:HamiltonianArchitecture}, ::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}, <:TrajectoryData}) = SEulerA()
default_method(::AbstractNeuralNetwork{<:HamiltonianArchitecture}, ::TrainingData{<:DataSymbol{<:DerivativePhaseSpaceSymbol}}) = ExactHnn()

default_method(::AbstractNeuralNetwork{<:SympNet}, ::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}, TrajectoryData}) = BasicSympNet()

default_method(::AbstractNeuralNetwork{<:LagrangianNeuralNetwork}, ::TrainingData{<:DataSymbol{<:PositionSymbol}, <:TrajectoryData}) = VariaMidPoint()
default_method(::AbstractNeuralNetwork{<:LagrangianNeuralNetwork}, ::TrainingData{<:DataSymbol{<:PosVeloAccSymbol}, <:SampledData} ) = ExactLnn()



