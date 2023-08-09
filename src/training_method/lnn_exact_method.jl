struct LnnExactMethod <: LnnTrainingMethod end

ExactLnn(;sqdist = sqeuclidean) = TrainingMethod{LnnExactMethod, PosVeloAccSymbol, SampledData, typeof(sqdist)}(sqdist)

function loss_single(::TrainingMethod{LnnExactMethod}, nn::NeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, q̇ₙ, q̈ₙ, params = nn.params)
    abs(sum(∇q∇q̇L(nn,qₙ, q̇ₙ, params)))  #inv(∇q̇∇q̇L(nn, qₙ, q̇ₙ, params))*(∇qL(nn, qₙ, q̇ₙ, params) - ∇q∇q̇L(nn, qₙ, q̇ₙ, params))
end

loss(ti::TrainingMethod{LnnExactMethod}, nn::NeuralNetwork{<:LagrangianNeuralNetwork}, data::TrainingData{<:DataSymbol{<:PosVeloAccSymbol}}, index_batch = eachindex(ti, data), params = nn.params) =
mapreduce((args...)->loss_single(ti, nn, get_data(data, :q,args...), get_data(data, :q̇, args...), get_data(data, :q̈, args...), params), +, index_batch)

