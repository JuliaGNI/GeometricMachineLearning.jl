struct BasicSympNetMethod <: SympNetTrainingMethod end

BasicSympNet(;sqdist = sqeuclidean) = TrainingMethod{BasicSympNetMethod, PhaseSpaceSymbol, TrajectoryData, typeof(sqdist)}(sqdist)

function loss_single(::TrainingMethod{BasicSympNetMethod}, nn::AbstractNeuralNetwork{<:SympNet}, qₙ, pₙ, qₙ₊₁, pₙ₊₁, params = nn.params)
    q̃ₙ₊₁,p̃ₙ₊₁ = nn([qₙ...,pₙ...],params)
    sqeuclidean(q̃ₙ₊₁,qₙ₊₁) + sqeuclidean(p̃ₙ₊₁,pₙ₊₁)
end

get_loss(::TrainingMethod{<:BasicSympNetMethod}, ::AbstractNeuralNetwork{<:SympNet}, data::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}}, args) = 
(Zygote.ignore_derivative(get_data(data,:q, args...)), Zygote.ignore_derivative(get_data(data,:p, args...)), Zygote.ignore_derivative(get_data(data,:q, next(args...)...)), Zygote.ignore_derivative(get_data(data,:p, next(args...)...)))

loss(ti::TrainingMethod{<:BasicSympNetMethod}, nn::AbstractNeuralNetwork{<:SympNet}, data::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}}, index_batch = eachindex(ti, data), params = nn.params) = 
mapreduce(args->loss_single(Zygote.ignore_derivative(ti), nn, get_loss(ti, nn, data, args)..., params),+, index_batch)
min_length_batch(::BasicSympNetMethod) = 2

