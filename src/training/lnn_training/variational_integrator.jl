abstract type VariationalIntegrator <: LnnTrainingIntegrator end
struct VariationalMidPointIntegrator <: VariationalIntegrator end
struct VariationalTrapezIntegrator <: VariationalIntegrator end

VariaMidPoint(;sqdist = sqeuclidean) = TrainingIntegrator{VariationalMidPointIntegrator, PositionSymbol, TrajectoryData, typeof(sqdist)}(sqdist)

# discrete langrangian
discrete_lagrangian(::TrainingIntegrator{VariationalMidPointIntegrator}, nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, qₙ₊₁, Δt, params = nn.params) =  nn([(qₙ₊₁+qₙ)/2..., (qₙ₊₁-qₙ)/Δt...], params)

# gradient of discrete Lagrangian
DL(discrete_lagrangian, ti::TrainingIntegrator{<:VariationalIntegrator}, nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, qₙ₊₁, Δt,  params = nn.params) = Zygote.gradient((qₙ,qₙ₊₁)->discrete_lagrangian(ti, nn, qₙ, qₙ₊₁, Δt, params), qₙ, qₙ₊₁)
DL₁(discrete_lagrangian, ti::TrainingIntegrator{<:VariationalIntegrator}, nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, qₙ₊₁, Δt, params = nn.params) = DL(discrete_lagrangian, ti, nn, qₙ, qₙ₊₁, Δt, params)[1:length(qₙ)]
DL₂(discrete_lagrangian, ti::TrainingIntegrator{<:VariationalIntegrator}, nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, qₙ₊₁, Δt, params = nn.params) = DL(discrete_lagrangian, ti, nn, qₙ, qₙ₊₁, Δt, params)[1+length(qₙ):end]


function loss_single(ti::TrainingIntegrator{<:VariationalIntegrator}, nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, qₙ₊₁, qₙ₊₂, Δt, params = nn.params)
    DL1 = DL₁(discrete_lagrangian, ti, nn, qₙ₊₁, qₙ₊₂, Δt, params)
    DL2 = DL₂(discrete_lagrangian, ti, nn, qₙ, qₙ₊₁, Δt,params)
    sqeuclidean(DL1,-DL2)
end

loss(ti::TrainingIntegrator{VariationalMidPointIntegrator}, nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, data::TrainingData{<:DataSymbol{<:PositionSymbol}}, index_batch = eachindex(ti, data), params = nn.params) =
mapreduce(args->loss_single(ti, nn, Zygote.ignore(get_data(data,:q, args...)), Zygote.ignore(get_data(data,:q, next(args...)...)), Zygote.ignore(get_data(data,:q,next(next(args...)...)...)), Zygote.ignore(get_Δt(data)), params), +, index_batch)

min_length_batch(::VariationalIntegrator) = 3
