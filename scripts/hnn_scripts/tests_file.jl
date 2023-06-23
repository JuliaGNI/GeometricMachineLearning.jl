
# using Profile
using GeometricMachineLearning

# this include the scripts using GeometricMachineLearning
include("hnn_script.jl")
include("../lnn_scripts/lnn_script.jl")
include("../sympnet_script.jl")

# this contains the functions for generating the training data
include("../data_problem.jl")

# this include the macro @testerror ans @testerror set for testing the code
include("../macro_test.jl")


#Data HNN with target
Data,Target = get_HNN_data(:pendulum)

Get_Data = Dict(
    :nb_points => Data -> length(Data),
    :q => (Data,n) -> Data[n][1],
    :p => (Data,n) -> Data[n][2]
)
pdata = DataSampled(Data, Get_Data)

Get_Target = Dict(
    :q̇ => (Target,n) -> Target[n][1],
    :ṗ => (Target,n) -> Target[n][2],
)

data = DataTarget(pdata, Target, Get_Target)

#Data multiple trajectory 
Data = get_multiple_trajectory_structure(:pendulum; n_trajectory = 2, n_points = 3, tstep = 0.1, qmin = -1.2, pmin = -1.2, qmax = 1.2, pmax = 1.2)

Get_Data = Dict(
    :Δt => Data -> Data.Δt,
    :nb_trajectory => Data -> Data.nb_trajectory,
    :length_trajectory => (Data,i) -> Data.data[Symbol("Trajectory_"*string(i))][:len],
    :q => (Data,i,n) -> Data.data[Symbol("Trajectory_"*string(i))][:data][n][1],
    :p => (Data,i,n) -> Data.data[Symbol("Trajectory_"*string(i))][:data][n][2],
)
data2 = DataTrajectory(Data, Get_Data)

#Data LNN with target
Data, Target = get_LNN_data(:pendulum)

Get_Data = Dict(
    :nb_points => Data -> length(Data),
    :q => (Data,n) -> Data[n][1][1],
    :q̇ => (Data,n) -> Data[n][2][1]
)
pdata = DataSampled(Data, Get_Data)

Get_Target = Dict(
    :q̈ => (Target,n) -> Target[n][1],
)

data3 = DataTarget(pdata, Target, Get_Target)

#Data LNN without target
Data = get_multiple_trajectory_structure(:pendulum; n_trajectory = 2, n_points = 3, tstep = 0.1, qmin = -1.2, pmin = -1.2, qmax = 1.2, pmax = 1.2)

Get_Data = Dict(
    :Δt => Data -> Data.Δt,
    :nb_trajectory => Data -> Data.nb_trajectory,
    :length_trajectory => (Data,i) -> Data.data[Symbol("Trajectory_"*string(i))][:len],
    :q => (Data,i,n) -> Data.data[Symbol("Trajectory_"*string(i))][:data][n][1],
)
data4 = DataTrajectory(Data, Get_Data)


@testseterrors begin

    @testerror HNN ExactHnn() data :pendulum MomentumOptimizer()
    @testerror HNN ExactHnn() data :pendulum AdamOptimizer()
    @testerror HNN ExactHnn() data :pendulum GradientOptimizer()

    @testerror HNN SEulerA() data2 :pendulum MomentumOptimizer()
    @testerror HNN SEulerA() data2 :pendulum AdamOptimizer()
    @testerror HNN SEulerA() data2 :pendulum GradientOptimizer()

    @testerror SYMPNET BasicSympNet() data2 :pendulum MomentumOptimizer()
    @testerror SYMPNET BasicSympNet() data2 :pendulum AdamOptimizer()
    @testerror SYMPNET BasicSympNet() data2 :pendulum GradientOptimizer()

    @testerror LNN ExactLnn() data3 :pendulum MomentumOptimizer()
    @testerror LNN ExactLnn() data3 :pendulum AdamOptimizer()
    @testerror LNN ExactLnn() data3 :pendulum GradientOptimizer()

    @testerror LNN VariaMidPoint() data4 :pendulum MomentumOptimizer()
    @testerror LNN VariaMidPoint() data4 :pendulum AdamOptimizer()
    @testerror LNN VariaMidPoint() data4 :pendulum GradientOptimizer()

end



