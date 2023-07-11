#=
    This file matches default integrator depending on the shape of data, its symbols, and the architecture of the neural network.
=#

default_integrator(::AbstractBackend, ::AbstractTrainingData) = throw(ArgumentError("Mismatch between the shape of data and the neural networks used to provide a default integrator for training"))

default_integrator(::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, ::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}, <:TrajectoryData}) = SEulerA()
default_integrator(::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, ::TrainingData{<:DataSymbol{<:DerivativePhaseSpaceSymbol}}) = ExactHnn()

default_integrator(::LuxNeuralNetwork{<:SympNet}, ::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}, TrajectoryData}) = BasicSympNet()

default_integrator(::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, ::TrainingData{<:DataSymbol{<:PositionSymbol}, <:TrajectoryData}) = VariaMidPoint()
default_integrator(::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, ::TrainingData{<:DataSymbol{<:PosVeloAccSymbol}, <:SampledData} ) = LnnExactIntegrator()



