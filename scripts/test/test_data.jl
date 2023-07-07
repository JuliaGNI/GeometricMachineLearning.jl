using GeometricMachineLearning
using Test




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


# Test for DataSymbol

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

@test symbol(DataSymbol(keys2), DataSymbol(keys1)) == (:p,)
@test symbol(DataSymbol(keys3), DataSymbol(keys1)) == (:p, :q̇, :q̈)


# Test for DataTraining

Data = (Trajectory1 =  ([0.0 0.0 0.0], [[0.0 0.0 0.0]]), Trajectory2 = ([0.2 0.5 0.7], [[0.7 0.8 0.9]]))
get_Data = Dict(
    :shape => TrajectoryData,
    :nb_trajectory => Data -> length(Data),
    :length_trajectory => (Data,i) -> length(Data[Symbol("Training"*String(i))][1])
    :Δt => Data -> 0.1
    :q => (Data,i,n) -> Data[Symbol("Training"*String(i))][1][n],
    :p => (Data,i,n) -> Data[Symbol("Training"*String(i))][2][n],

)
training_data = TrainingData(data, get_data)

@test problem(training_data)        == Unknownproblem
@test typeof(shape(training_data))  == TrajectroyData
@test type(symbols(data))           == PhaseSpaceSymbol
@test symbols(training_data)        == (:q,:p)
@test dim(training_data)            == 1
@test noisemaker(training_data)     == NothingFunction()    

@test get_Δt(training_data)                == 0.1
@test get_nb_trajectory(training_data)     == 2
@test get_length_trajectory(training_data) == [3,3]
@test get_nb_point(training_data)         === nothing
@test get_data(training_data, :p, 2,1)     == 0.7

@test eachindex(training_data) == [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]


sampled_data = reshape_intoSampledData!(training_data)

@test problem(sampled_data)         == Unknownproblem
@test typeof(shape(sampled_data))   == SampledData
@test type(symbols(sampled_data))   == PhaseSpaceSymbol
@test symbols(sampled_data)         == (:q,:p)
@test dim(sampled_data)             == 1
@test noisemaker(sampled_data)      == NothingFunction()    

@test get_Δt(sampled_data)                === nothing
@test get_nb_trajectory(sampled_data)     === nothing
@test get_length_trajectory(sampled_data) === nothing
@test get_nb_point(sampled_data)           == 6
@test get_data(sampled_data, :p, 4)        == 0.7

@test eachindex(sampled_data) == [1,2,3,4,5,6]


reduced_data = reduce_symbols(sampled_data, DataSymbol((:q,)))

@test problem(reduced_data)         == Unknownproblem
@test typeof(shape(reduced_data))   == SampledData
@test type(symbols(reduced_data))   == PositionSymbol
@test symbols(reduced_data)         == (:q,)
@test dim(reduced_data)             == 1
@test noisemaker(reduced_data)      == NothingFunction()    

@test Tuple(keys(get(reduced_data)) == (:q,)


# Test Batch

@test get_batch(training_data, (2))

# Test Mathcing



# Test for TrainingSet

training_set1 = TrainingSet(nn, training_parameters, training_data)

@test nn(training_set1)

# Test for EnsembleTraining

ensemble_training1 = 

