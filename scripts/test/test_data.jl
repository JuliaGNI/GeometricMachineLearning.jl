using GeometricMachineLearning


# Test for TrainingParameters

nruns = 10
method = ExactHnn()
mopt = GradientOptimizer()
bs = 10

training_parameters = TrainingParameters(nruns, method, mopt; batch_size = bs)

@test nruns(training_parameters) == nruns
@test method(training_parameters) == method
@test opt(training_parameters) == mopt
@test batchsize(training_parameters) == bs


# Test for DataShape



# Test for DataTraining



# Test for TrainingSet

training_set1 = 

# Test for EnsembleTraining

ensemble_training1 = 

