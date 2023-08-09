abstract type VariationalMethod <: LnnTrainingMethod end
struct VariationalMidPointMethod <: VariationalMethod end
struct VariationalTrapezMethod <: VariationalMethod end

VariaMidPoint(;sqdist = sqeuclidean) = TrainingMethod{VariationalMidPointMethod, PositionSymbol, TrajectoryData, typeof(sqdist)}(sqdist)

# discrete langrangian
discrete_lagrangian(::TrainingMethod{VariationalMidPointMethod}, nn::NeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, qₙ₊₁, Δt, params = nn.params) =  nn([(qₙ₊₁+qₙ)/2..., (qₙ₊₁-qₙ)/Δt...], params)

# gradient of discrete Lagrangian
DL(ti::TrainingMethod{<:VariationalMethod}, nn::NeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, qₙ₊₁, Δt,  params = nn.params) = Zygote.gradient((qₙ,qₙ₊₁)->discrete_lagrangian(ti, nn, qₙ, qₙ₊₁, Δt, params), qₙ, qₙ₊₁)
DL₁(ti::TrainingMethod{<:VariationalMethod}, nn::NeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, qₙ₊₁, Δt, params = nn.params) = DL(ti, nn, qₙ, qₙ₊₁, Δt, params)[1:length(qₙ)]
DL₂(ti::TrainingMethod{<:VariationalMethod}, nn::NeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, qₙ₊₁, Δt, params = nn.params) = DL(ti, nn, qₙ, qₙ₊₁, Δt, params)[1+length(qₙ):end]


function loss_single(ti::TrainingMethod{<:VariationalMethod}, nn::NeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, qₙ₊₁, qₙ₊₂, Δt, params = nn.params)
    DL1 = DL₁(ti, nn, qₙ₊₁, qₙ₊₂, Δt, params)
    DL2 = DL₂(ti, nn, qₙ, qₙ₊₁, Δt,params)
    sqeuclidean(DL1,-DL2)
end

loss(ti::TrainingMethod{VariationalMidPointMethod}, nn::NeuralNetwork{<:LagrangianNeuralNetwork}, data::TrainingData{<:DataSymbol{<:PositionSymbol}}, index_batch = eachindex(ti, data), params = nn.params) =
mapreduce(args->loss_single(ti, nn, Zygote.ignore(get_data(data,:q, args...)), Zygote.ignore(get_data(data,:q, next(args...)...)), Zygote.ignore(get_data(data,:q,next(next(args...)...)...)), Zygote.ignore(get_Δt(data)), params), +, index_batch)

min_length_batch(::VariationalMethod) = 3
