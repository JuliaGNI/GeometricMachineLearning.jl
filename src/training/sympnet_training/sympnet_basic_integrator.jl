struct BasicSympNetIntegrator <: SympNetTrainingIntegrator end

BasicSympNet(;sqdist = sqeuclidean) = TrainingIntegrator{BasicSympNetIntegrator, PhaseSpaceSymbol, TrajectoryData, typeof(sqdist)}(sqdist)

function loss_single(::TrainingIntegrator{BasicSympNetIntegrator}, nn::LuxNeuralNetwork{<:SympNet}, qₙ, pₙ, qₙ₊₁, pₙ₊₁, params = nn.params)
    q̃ₙ₊₁,p̃ₙ₊₁ = nn([qₙ...,pₙ...],params)
    sqeuclidean(q̃ₙ₊₁,qₙ₊₁) + sqeuclidean(p̃ₙ₊₁,pₙ₊₁)
end

loss(ti::TrainingIntegrator{BasicSympNetIntegrator}, nn::LuxNeuralNetwork{<:SympNet}, data::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}}, index_batch = eachindex(ti, data), params = nn.params) =
mapreduce(args->loss_single(ti, nn, Zygote.ignore(get_data(data,:q, args...)), Zygote.ignore(get_data(data,:p, args...)), Zygote.ignore(get_data(data,:q, next(args...)...)), Zygote.ignore(get_data(data,:p, next(args...)...)), params), +, index_batch)

min_length_batch(::BasicSympNetIntegrator) = 2

