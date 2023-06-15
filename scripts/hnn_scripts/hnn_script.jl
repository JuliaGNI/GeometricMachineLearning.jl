# using Profile
using GeometricMachineLearning

# this contains the functions for generating the training data
include("../data_problem.jl")


function HNN_TARGET(;nameproblem::Symbol = :pendulum, opt =  MomentumOptimizer(1e-3,0.5), integrator::Hnn_training_integrator = )
    
    _, n_dim = dict_problem_H[nameproblem]

    # layer dimension/width
    const ld = 5

    # hidden layers
    const ln = 3

    # number of inputs/dimension of system
    const ninput = 2*n_dim

    # number of training runs
    const nruns = 10

    # create HNN
    hnn = HamiltonianNeuralNetwork(ninput; nhidden = ln, width = ld)

    # create Lux network
    nn = NeuralNetwork(hnn, LuxBackend())

    # get data set
    println("Begin generating data")
    
    
    #nameproblem = :pendulum
    #data = get_multiple_trajectory_structure(nameproblem; n_trajectory = 10, n_points = 1000, tstep = 0.1, qmin = -1.2, pmin = -1.2, qmax = 1.2, pmax = 1.2)
    #data_t = get_multiple_trajectory_structure_with_target(nameproblem; n_trajectory = 10, n_points = 100, tstep = 0.1, qmin = -1.2, pmin = -1.2, qmax = 1.2, pmax = 1.2)

    Data,Target = get_HNN_data(nameproblem)

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

    println("End generating data")

    #training method
    hti = ExactIntegrator()

    # perform training (returns array that contains the total loss for each training step)
    total_loss = train!(nn, opt, data; ntraining = nruns, hti)

end


# plot results
include("plots.jl")
plot_hnn(H, nn, total_loss; filename="hnn_pendulum.png", xmin=-1.2, xmax=+1.2, ymin=-1.2, ymax=+1.2)



