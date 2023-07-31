#=
    This file matches default integrator depending on the shape of data, its symbols, and the architecture of the neural network.
=#

default_integrator(::AbstractBackend, ::AbstractTrainingData) = throw(ArgumentError("Mismatch between the shape of data and the neural networks used to provide a default integrator for training"))

default_integrator(::NeuralNetwork{<:HamiltonianNeuralNetwork}, ::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}, <:TrajectoryData}) = SEulerA()
default_integrator(::NeuralNetwork{<:HamiltonianNeuralNetwork}, ::TrainingData{<:DataSymbol{<:DerivativePhaseSpaceSymbol}}) = ExactHnn()

default_integrator(::NeuralNetwork{<:SympNet}, ::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}, TrajectoryData}) = BasicSympNet()

default_integrator(::NeuralNetwork{<:LagrangianNeuralNetwork}, ::TrainingData{<:DataSymbol{<:PositionSymbol}, <:TrajectoryData}) = VariaMidPoint()
default_integrator(::NeuralNetwork{<:LagrangianNeuralNetwork}, ::TrainingData{<:DataSymbol{<:PosVeloAccSymbol}, <:SampledData} ) = ExactLnn()



