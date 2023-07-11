using GeometricMachineLearning
using Test

include("macro_testerror.jl")
include("generation_of_data.jl")

#########################################
# Test for creation of TrainingIntegrator
#########################################

exacthnn = ExactHnn()

@test type(exacthnn)    == HnnExactIntegrator
@test symbols(exacthnn) == DerivativePhaseSpaceSymbol
@test shape(exacthnn)   == SampledData
@test min_length_batch(exacthnn) == 1

sympeuler = SEuler()

@test type(sympeuler)    == SymplecticEulerA
@test symbols(sympeuler) == PhaseSpaceSymbol
@test shape(sympeuler)   == TrajectoryData
@test min_length_batch(sympeuler) == 2

msympnet = BasicSympNet()

@test type(msympnet)    == BasicSympNetIntegrator
@test symbols(msympnet) == PhaseSpaceSymbol
@test shape(msympnet)   == TrajectoryData
@test min_length_batch(msympnet) == 2

exactlnn = ExactLnn()

@test type(exactlnn)    == LnnExactIntegrator
@test symbols(exactlnn) == PosVeloAccSymbol
@test shape(exactlnn)   == SampledData
@test min_length_batch(exactlnn) == 1

midpointlnn = VariaMidPoint()

@test type(midpointlnn)    == VariationalMidPointIntegrator
@test symbols(midpointlnn) == PositionSymbol
@test shape(midpointlnn)   == TrajectoryData
@test min_length_batch(midpointlnn) == 3

#########################################
# Test for default_integrator
#########################################

hnn = NeuralNetwork(HamiltonianNeuralNetwork(2))
sympnet = NeuralNetwork(GSympNet(2))
lnn = NeuralNetwork(LagrangianNeuralNetwork(2))

@testerror type(default_integrator(sympnet, tra_pos_data))
@test type(default_integrator(hnn, tra_ps_data)) == SymplecticEulerA
@test type(default_integrator(hnn, sam_dps_data)) == HnnExactIntegrator
@test type(default_integrator(sympnet, tra_ps_data)) == BasicSympNetIntegrator
@test type(default_integrator(lnn, tra_pos_data)) == VariationalMidPointIntegrator
#@test type(default_integrator(lnn, sam_accposvel_data)) == LnnExactIntegrator


#########################################
# Test for matching
#########################################





