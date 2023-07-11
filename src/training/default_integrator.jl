#=
    This file matches default integrator depending on the shape of data, its symbols, and the architecture of the neural network.
=#

default_integrator(::AbstractBackend, ::AbstractTrainingData) = throw(ArgumentError("Mismatch between the shape of data and the neural networks used to provide a default integrator for training"))

default_integrator(::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, ::TrainingData{<:PhaseSpaceSymbol, <:TrajectoryData}) = SEulerA()
default_integrator(::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, ::TrainingData{<:DerivativePhaseSpaceSymbol, <:TrajectoryData}) = ExactHnn()
default_integrator(::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, ::TrainingData{<:DerivativePhaseSpaceSymbol, <:SampledData}) = ExactHnn()

default_integrator(::LuxNeuralNetwork{<:SympNet}, ::TrainingData{<:PhaseSpaceSymbol, TrajectoryData}) = BasicSympNet()

default_integrator(::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, ::TrainingData{<:PositionSymbol, <:TrajectoryData}) = VariationalMidPointIntegrator()
default_integrator(::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, ::TrainingData{<:PosVeloAccSymbol, <:TrajectoryData} ) = LnnExactIntegrator()
default_integrator(::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, ::TrainingData{<:PosVeloAccSymbol, <:SampledData} ) = LnnExactIntegrator()



