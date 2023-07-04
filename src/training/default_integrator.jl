
default_integrator(::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, ::TrainingData{T,TrajectoryData} where T) = SymplecticEulerA()
#default_integrator(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, data::DataTarget) = HnnExactIntegrator()

#default_integrator(nn::LuxNeuralNetwork{<:SympNet}, data::DataTrajectory) = BasicSympNetIntegrator()


#default_integrator(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, data::DataTarget) = LnnExactIntegrator()
#default_integrator(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, data::DataTrajectory) = VariationalMidPointIntegrator()



