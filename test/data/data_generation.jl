##############################################################
# Generation of Data{PositionSymbol, Trajectory}
##############################################################

Data = (Trajectory1 =  [0.0 0.0 0.0], Trajectory2 = [0.2 0.5 0.7])
get_Data = Dict(
    :shape => TrajectoryData,
    :nb_trajectory => Data -> length(Data),
    :length_trajectory => (Data,i) -> length(Data[Symbol("Trajectory"*string(i))]),
    :Δt => Data -> 0.1,
    :q => (Data,i,n) -> Data[Symbol("Trajectory"*string(i))][n],
)
tra_pos_data= TrainingData(Data, get_Data)

##############################################################
# Generation of Data{PosVeloSymbol, Trajectory}
##############################################################

Data = (Trajectory1 =  ([0.0 0.0 0.0], [0.0 0.0 0.0]), Trajectory2 = ([0.2 0.5 0.7], [0.7 0.8 0.9]))
get_Data = Dict(
    :shape => TrajectoryData,
    :nb_trajectory => Data -> length(Data),
    :length_trajectory => (Data,i) -> length(Data[Symbol("Trajectory"*string(i))][1]),
    :Δt => Data -> 0.1,
    :q => (Data,i,n) -> Data[Symbol("Trajectory"*string(i))][1][n],
    :q̇ => (Data,i,n) -> Data[Symbol("Trajectory"*string(i))][2][n]
)
tra_posvelo_data = TrainingData(Data, get_Data)


##############################################################
# Generation of Data{PosVeloAccSymbol, Sampled}
##############################################################

Data = ([0.1 0.2 0.3 0.4 0.5 0.6], [0.2 0.4 0.6 0.8 1.0 1.2], [1.0 1.0 1.0 1.0 1.0 1.0])
get_Data = Dict(
    :shape => SampledData,
    :nb_points=> Data -> length(Data[1]),
    :q => (Data,n) -> Data[1][n],
    :q̇ => (Data,n) -> Data[2][n],
    :q̈ => (Data,n) -> Data[3][n]
)
sam_accposvel_data = TrainingData(Data, get_Data)



##############################################################
# Generation of Data{PhaseSpaceSymbol, Trajectory}
##############################################################

Data = ([0.1 0.2 0.3 0.4 0.5 0.6], [0.2 0.4 0.6 0.8 1.0 1.2])
get_Data = Dict(
    :shape => SampledData,
    :nb_points=> Data -> length(Data[1]),
    :q => (Data,n) -> Data[1][n],
    :p => (Data,n) -> Data[2][n]
)
sam_ps_data = TrainingData(Data, get_Data)

Data_traps = (Trajectory1 =  ([0.0 0.0 0.0], [0.0 0.0 0.0]), Trajectory2 = ([0.2 0.5 0.7], [0.7 0.8 0.9]))
get_Data_traps = Dict(
    :shape => TrajectoryData,
    :nb_trajectory => Data -> length(Data),
    :length_trajectory => (Data,i) -> length(Data[Symbol("Trajectory"*string(i))][1]),
    :Δt => Data -> 0.1,
    :q => (Data,i,n) -> Data[Symbol("Trajectory"*string(i))][1][n],
    :p => (Data,i,n) -> Data[Symbol("Trajectory"*string(i))][2][n],
)
tra_ps_data = TrainingData(Data_traps, get_Data_traps)

##############################################################
# Generation of Data{DerivativePhaseSpaceSymbol, Sampled}
##############################################################

Data = ([0.1 0.2 0.3 0.4 0.5 0.6], [0.2 0.4 0.6 0.8 1.0 1.2], [1.0 1.0 1.0 1.0 1.0 1.0], [1.0 1.0 1.0 1.0 1.0 1.0])
get_Data = Dict(
    :shape => SampledData,
    :nb_points=> Data -> length(Data[1]),
    :q => (Data,n) -> Data[1][n],
    :p => (Data,n) -> Data[2][n],
    :q̇ => (Data,n) -> Data[3][n],
    :ṗ => (Data,n) -> Data[4][n],
)
sam_dps_data = TrainingData(Data, get_Data)


Data = (Trajectory1 =  ([0.0 0.0 0.0], [0.0 0.0 0.0], [0.0 0.0 0.0], [0.0 0.0 0.0]), Trajectory2 = ([0.2 0.5 0.7], [0.7 0.8 0.9], [0.7 0.8 0.9], [0.2 0.5 0.7]))
get_Data = Dict(
    :shape => TrajectoryData,
    :nb_trajectory => Data -> length(Data),
    :length_trajectory => (Data,i) -> length(Data[Symbol("Trajectory"*string(i))][1]),
    :Δt => Data -> 0.1,
    :q => (Data,i,n) -> Data[Symbol("Trajectory"*string(i))][1][n],
    :p => (Data,i,n) -> Data[Symbol("Trajectory"*string(i))][2][n],
    :q̇ => (Data,i,n) -> Data[Symbol("Trajectory"*string(i))][3][n],
    :ṗ => (Data,i,n) -> Data[Symbol("Trajectory"*string(i))][4][n],
)
tra_dps_data = TrainingData(Data, get_Data)