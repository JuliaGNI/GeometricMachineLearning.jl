abstract type VariationalIntegrator <: LnnTrainingIntegrator end
struct VariationalMidPointIntegrator <: VariationalIntegrator end
struct VariationalTrapezIntegrator <: VariationalIntegrator end

VariaMidPoint(;sqdist = sqeuclidean) = TrainingIntegrator(VariationalMidPointIntegrator(); sqdist = sqdist)


# discrete langrangian
discrete_lagrangian(::VariationalMidPointIntegrator, nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, qₙ₊₁, Δt, params = nn.params) =  nn([(qₙ₊₁+qₙ)/2..., (qₙ₊₁-qₙ)/Δt...], params)

# gradient of discrete Lagrangian
DL(discrete_lagrangian, ti::VariationalIntegrator, nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, qₙ₊₁, Δt,  params = nn.params) = Zygote.gradient((qₙ,qₙ₊₁)->discrete_lagrangian(ti, nn, qₙ, qₙ₊₁, Δt, params), qₙ, qₙ₊₁)
DL₁(discrete_lagrangian, ti::VariationalIntegrator, nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, qₙ₊₁, Δt, params = nn.params) = DL(discrete_lagrangian, ti, nn, qₙ, qₙ₊₁, Δt, params)[1:length(qₙ)]
DL₂(discrete_lagrangian, ti::VariationalIntegrator, nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, qₙ₊₁, Δt, params = nn.params) = DL(discrete_lagrangian, ti, nn, qₙ, qₙ₊₁, Δt, params)[1+length(qₙ):end]


function loss_single(ti::VariationalIntegrator, nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, qₙ₊₁, qₙ₊₂, Δt, params = nn.params)
    DL1 = DL₁(discrete_lagrangian, ti, nn, qₙ₊₁, qₙ₊₂, Δt, params)
    DL2 = DL₂(discrete_lagrangian, ti, nn, qₙ, qₙ₊₁, Δt,params)
    sqeuclidean(DL1,-DL2)
end

loss(ti::VariationalMidPointIntegrator, nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, datat::DataTrajectory, index_batch = get_batch(datat), params = nn.params) =
mapreduce(x->loss_single(ti, nn, Zygote.ignore(get_data(datat,:q,x[1],x[2])), Zygote.ignore(get_data(datat,:q,x[1],x[2]+1)), Zygote.ignore(get_data(datat,:q,x[1],x[2]+2)), Zygote.ignore(get_Δt(datat)), params), +, index_batch)

required_key(::VariationalMidPointIntegrator) = (:q,)

min_length_batch(::VariationalIntegrator) = 3

data_goal(::VariationalIntegrator) = (test_data_trajectory,)