using GeometricMachineLearning
using Test

include("macro_testerror.jl")
include("data_generation.jl")

hnn = NeuralNetwork(HamiltonianNeuralNetwork(2), Float64)
sympnet = NeuralNetwork(GSympNet(2), Float64)
lnn = NeuralNetwork(LagrangianNeuralNetwork(2), Float64)

#########################################
# Test for creation of TrainingMethod
#########################################

exacthnn = ExactHnn()

@test type(exacthnn)    == HnnExactMethod
@test symbols(exacthnn) == DerivativePhaseSpaceSymbol
@test shape(exacthnn)   == SampledData
@test min_length_batch(exacthnn) == 1
@test typeof(loss(exacthnn, hnn, sam_dps_data)) <: Real
@test loss(exacthnn, hnn, sam_dps_data) > 0

sympeuler = SEuler()

@test type(sympeuler)    == SymplecticEulerA
@test symbols(sympeuler) == PhaseSpaceSymbol
@test shape(sympeuler)   == TrajectoryData
@test min_length_batch(sympeuler) == 2
@test typeof(loss(sympeuler, hnn, tra_ps_data)) <: Real
@test loss(sympeuler, hnn, tra_ps_data) > 0

msympnet = BasicSympNet()

@test type(msympnet)    == BasicSympNetMethod
@test symbols(msympnet) == PhaseSpaceSymbol
@test shape(msympnet)   == TrajectoryData
@test min_length_batch(msympnet) == 2
@test typeof(loss(msympnet, sympnet, tra_ps_data)) <: Real
@test loss(msympnet, sympnet, tra_ps_data) > 0

exactlnn = ExactLnn()

@test type(exactlnn)    == LnnExactMethod
@test symbols(exactlnn) == PosVeloAccSymbol
@test shape(exactlnn)   == SampledData
@test min_length_batch(exactlnn) == 1
@test typeof(loss(exactlnn, lnn, sam_accposvel_data)) <: Real
@test loss(exactlnn, lnn, sam_accposvel_data) > 0

midpointlnn = VariaMidPoint()

@test type(midpointlnn)    == VariationalMidPointMethod
@test symbols(midpointlnn) == PositionSymbol
@test shape(midpointlnn)   == TrajectoryData
@test min_length_batch(midpointlnn) == 3
#@test typeof(loss(midpointlnn, lnn, tra_pos_data)) <: Real
#@test loss(midpointlnn, lnn, tra_pos_data) > 0


#########################################
# Test for default_Method
#########################################

@testerror type(default_Method(sympnet, tra_pos_data))
@test type(default_method(hnn, tra_ps_data)) == SymplecticEulerA
@test type(default_method(hnn, sam_dps_data)) == HnnExactMethod
@test type(default_method(sympnet, tra_ps_data)) == BasicSympNetMethod
@test type(default_method(lnn, tra_pos_data)) == VariationalMidPointMethod
@test type(default_method(lnn, sam_accposvel_data)) == LnnExactMethod
