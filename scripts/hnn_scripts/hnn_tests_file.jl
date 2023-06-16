
# using Profile
using GeometricMachineLearning

# this include the scripts using GeometricMachineLearning
include("hnn_script.jl")

# this contains the functions for generating the training data
include("../data_problem.jl")

# this include the macro @testerror ans @testerror set for testing the code
include("../macro_test.jl")



Data,Target = get_HNN_data(:pendulum)

Get_Data = Dict(
    :nb_points => Data -> length(Data),
    :q => (Data,n) -> Data[n][1],
    :p => (Data,n) -> Data[n][2]
)
pdata = data_sampled(Data, Get_Data)

Get_Target = Dict(
    :q̇ => (Target,n) -> Target[n][1],
    :ṗ => (Target,n) -> Target[n][2],
)

data = dataTarget(pdata, Target, Get_Target)


Data = get_multiple_trajectory_structure(:pendulum; n_trajectory = 2, n_points = 3, tstep = 0.1, qmin = -1.2, pmin = -1.2, qmax = 1.2, pmax = 1.2)

Get_Data = Dict(
    :Δt => Data -> Data.Δt,
    :nb_trajectory => Data -> Data.nb_trajectory,
    :length_trajectory => (Data,i) -> Data.data[Symbol("Trajectory_"*string(i))][:len],
    :q => (Data,i,n) -> Data.data[Symbol("Trajectory_"*string(i))][:data][n][1],
    :p => (Data,i,n) -> Data.data[Symbol("Trajectory_"*string(i))][:data][n][2],
)
data2 = data_trajectory(Data, Get_Data)

@testseterrors begin

    @testerror HNN ExactIntegrator() data :pendulum MomentumOptimizer()
    @testerror HNN ExactIntegrator() data :pendulum AdamOptimizer()
    @testerror HNN ExactIntegrator() data :pendulum GradientOptimizer()

    #@testerror HNN SEuler() data2 :pendulum MomentumOptimizer()
    #@testerror HNN SEuler() data2 :pendulum AdamOptimizer()
    #@testerror HNN SEuler() data2 :pendulum GradientOptimizer()


end



