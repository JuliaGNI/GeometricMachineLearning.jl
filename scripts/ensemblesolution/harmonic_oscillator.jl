using GeometricMachineLearning
using GeometricSolutions
using Test

using GeometricProblems.HarmonicOscillator
using GeometricProblems.HarmonicOscillator: harmonic_oscillator_hode_ensemble


#creer l'object ensemble solution
ensemble_solution = harmonic_oscillator_hode_ensemble()

#creer le training data associé
data_training = TrainingData(ensemble_solution)

@test problem(training_data)               == UnknownProblem()
@test typeof(shape(training_data))         == TrajectoryData
@test type(data_symbols(training_data))    == PhaseSpaceSymbol
@test symbols(training_data)               == (:q,:p)
@test dim(training_data)                   == 2
@test noisemaker(training_data)            == NothingFunction()    

@test get_Δt(training_data)                == 0.1
@test get_nb_trajectory(training_data)     == 10
@test get_length_trajectory(training_data) == 11
@test get_nb_point(training_data)         === nothing

@test Tuple(keys(GeometricMachineLearning.get(training_data))) ==(:q, :p)







