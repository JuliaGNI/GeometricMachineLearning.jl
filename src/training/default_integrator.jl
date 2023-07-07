
default_integrator(::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, ::TrainingData{T,TrajectoryData} where T) = SEulerA()
default_integrator(::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, ::TrainingData{T,SampledData} where T) = ExactHnn()
default_integrator(::LuxNeuralNetwork{<:SympNet}, ::TrainingData{T,TrajectoryData} where T) = BasicSympNet()


#default_integrator(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, data::DataTarget) = LnnExactIntegrator()
#default_integrator(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, data::DataTrajectory) = VariationalMidPointIntegrator()



