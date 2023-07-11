using GeometricMachineLearning
using Test

#########################################
# Test for creation of TrainingIntegrator
#########################################

exacthnn = ExactHnn()

@test type(exacthnn)    == HnnExactIntegrator
@test symbols(exacthnn) == DerivativePhaseSpaceSymbol
@test shape(exacthnn)   == SampledData
