struct HnnExactMethod <: HnnTrainingMethod end

ExactHnn(;sqdist = sqeuclidean) = TrainingMethod{HnnExactMethod, DerivativePhaseSpaceSymbol, SampledData, typeof(sqdist)}(sqdist)

function loss_single(::TrainingMethod{HnnExactMethod}, nn::NeuralNetwork{<:HamiltonianNeuralNetwork}, qₙ, pₙ, q̇ₙ, ṗₙ, params = nn.params)
    dH = vectorfield(nn, [qₙ...,pₙ...], params)
    sqeuclidean(dH[1],q̇ₙ) + sqeuclidean(dH[2],ṗₙ)
end

get_loss(::TrainingMethod{HnnExactMethod}, ::AbstractNeuralNetwork{<:HamiltonianNeuralNetwork}, data::TrainingData{<:DataSymbol{<:DerivativePhaseSpaceSymbol}}, args) = (Zygote.ignore_derivatives(get_data(data,:q, args...)), Zygote.ignore_derivatives(get_data(data,:p, args...)), Zygote.ignore_derivatives(get_data(data,:q̇, args...)), Zygote.ignore_derivatives(get_data(data,:ṗ, args...)))

loss(ti::TrainingMethod{HnnExactMethod}, nn::NeuralNetwork{<:HamiltonianNeuralNetwork}, data::TrainingData{<:DataSymbol{<:DerivativePhaseSpaceSymbol}}, index_batch = eachindex(ti, data), params = nn.params) = 
mapreduce(args->loss_single(Zygote.ignore_derivatives(ti), nn, get_loss(ti, nn, data, args)..., params), +, index_batch)
