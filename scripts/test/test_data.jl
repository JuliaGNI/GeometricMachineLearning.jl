using GeometricMachineLearning 
using Test

#########################################
# Test for TrainingParameters
#########################################

nruns = 10
method = ExactHnn()
mopt = GradientOptimizer()
bs = 10

training_parameters = TrainingParameters(nruns, method, mopt; batch_size = bs)

@test GeometricMachineLearning.nruns(training_parameters) == nruns
@test GeometricMachineLearning.method(training_parameters) == method
@test GeometricMachineLearning.opt(training_parameters) == mopt
@test GeometricMachineLearning.batchsize(training_parameters) == bs

#########################################
# Test for DataSymbol
#########################################

keys1 = (:q,)
keys2 = (:q,:p)
keys3 = (:q,:p, :q̇, :ṗ)
keys4 = (:q, :q̇)
keys5 = (:q, :q̇, :q̈)
keys6 = (:q, :p, :s)

@test type(DataSymbol(keys1)) == PositionSymbol
@test type(DataSymbol(keys2)) == PhaseSpaceSymbol
@test type(DataSymbol(keys3)) == DerivativePhaseSpaceSymbol
@test type(DataSymbol(keys4)) == PosVeloSymbol
@test type(DataSymbol(keys5)) == PosVeloAccSymbol
@test type(DataSymbol(keys6)) == PhaseSpaceSymbol


@test can_reduce(DataSymbol(keys2), DataSymbol(keys1)) == true
@test can_reduce(DataSymbol(keys3), DataSymbol(keys1)) == true
@test can_reduce(DataSymbol(keys5), DataSymbol(keys2)) == false

@test symboldiff(DataSymbol(keys2), DataSymbol(keys1)) == (:p,)
@test symboldiff(DataSymbol(keys3), DataSymbol(keys1)) == (:p, :q̇, :ṗ)

#########################################
# Test for DataTraining
#########################################

Data = (Trajectory1 =  ([0.0 0.0 0.0], [0.0 0.0 0.0]), Trajectory2 = ([0.2 0.5 0.7], [0.7 0.8 0.9]))
get_Data = Dict(
    :shape => TrajectoryData,
    :nb_trajectory => Data -> length(Data),
    :length_trajectory => (Data,i) -> length(Data[Symbol("Trajectory"*string(i))][1]),
    :Δt => Data -> 0.1,
    :q => (Data,i,n) -> Data[Symbol("Trajectory"*string(i))][1][n],
    :p => (Data,i,n) -> Data[Symbol("Trajectory"*string(i))][2][n],
)
training_data = TrainingData(Data, get_Data)

@test problem(training_data)               == UnknownProblem()
@test typeof(shape(training_data))         == TrajectoryData
@test type(data_symbols(training_data))    == PhaseSpaceSymbol
@test symbols(training_data)               == (:q,:p)
@test dim(training_data)                   == 1
@test noisemaker(training_data)            == NothingFunction()    

@test get_Δt(training_data)                == 0.1
@test get_nb_trajectory(training_data)     == 2
@test get_length_trajectory(training_data) == [3,3]
@test get_nb_point(training_data)         === nothing
@test get_data(training_data, :p, 2,1)     == 0.7

@test eachindex(training_data) == [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]


sampled_data = reshape_intoSampledData(training_data)

@test problem(sampled_data)              == UnknownProblem()
@test typeof(shape(sampled_data))        == SampledData
@test type(data_symbols(sampled_data))   == PhaseSpaceSymbol
@test symbols(sampled_data)              == (:q,:p)
@test dim(sampled_data)                  == 1
@test noisemaker(sampled_data)           == NothingFunction()    

@test get_Δt(sampled_data)                === nothing
@test get_nb_trajectory(sampled_data)     === nothing
@test get_length_trajectory(sampled_data) === nothing
@test get_nb_point(sampled_data)           == 6
@test get_data(sampled_data, :p, 4)        == 0.7

@test eachindex(sampled_data) == [1,2,3,4,5,6]


reduced_data = reduce_symbols(sampled_data, DataSymbol((:q,)))

@test problem(reduced_data)              == UnknownProblem()
@test typeof(shape(reduced_data))        == SampledData
@test type(data_symbols(reduced_data))   == PositionSymbol
@test symbols(reduced_data)              == (:q,)
@test dim(reduced_data)                  == 1
@test noisemaker(reduced_data)           == NothingFunction()    

@test Tuple(keys(GeometricMachineLearning.get(reduced_data))) == (:q,)

#########################################
# Test Batch
#########################################

index_batch = get_batch(training_data, (2,2,2))
@test length(index_batch) == 2
for i in index_batch
    x,y = i 
    @test 1<= x <= get_nb_trajectory(training_data)
    @test 1<= y <= get_length_trajectory(training_data, x)
end

#default index batch ?

#########################################
# Test for TrainingSet
#########################################

hnn = HamiltonianNeuralNetwork(2; nhidden= 2, width = 5)
nn = NeuralNetwork(hnn, LuxBackend())

training_set1 = TrainingSet(nn, training_parameters, training_data)

@test GeometricMachineLearning.nn(training_set1) == nn
@test parameters(training_set1) == training_parameters
@test data(training_set1) == training_data

# Test for EnsembleTraining

ensemble_training1 = 

