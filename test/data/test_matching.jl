using GeometricMachineLearning 
using Test

include("../macro_testerror.jl")
include("data_generation.jl")

#########################################
# Test matching for shape
#########################################

exacthnn = ExactHnn()
sympeuler = SEuler()

@test matching(exacthnn, sam_dps_data) == sam_dps_data
@test matching(sympeuler, tra_ps_data) == tra_ps_data

match1 = matching(exacthnn, tra_dps_data)
@test typeof(shape(match1)) == SampledData
@test type(data_symbols(match1)) == type(data_symbols(tra_dps_data))

@testerror matching(sympeuler, sam_ps_data)

#########################################
# Test matching for symbols
#########################################

match2 = matching(sympeuler, tra_dps_data)
@test typeof(shape(match2)) == typeof(shape(tra_dps_data))
@test type(data_symbols(match2)) == PhaseSpaceSymbol


