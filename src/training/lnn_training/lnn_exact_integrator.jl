struct LnnExactIntegrator <: LnnTrainingIntegrator end

ExactLnn(;sqdist = sqeuclidean) = TrainingIntegrator{LnnExactIntegrator, PosVeloAccSymbol, SampledData, typeof(sqdist)}(sqdist)

function loss_single(::LnnExactIntegrator, nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, q̇ₙ, q̈ₙ, params = nn.params)
    sum(∇q∇q̇L(nn,qₙ, q̇ₙ, params))  #inv(∇q̇∇q̇L(nn, qₙ, q̇ₙ, params))*(∇qL(nn, qₙ, q̇ₙ, params) - ∇q∇q̇L(nn, qₙ, q̇ₙ, params))
end

loss(ti::LnnExactIntegrator, nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, data::TrainingData{DataSymbol{<:PosVeloAccSymbol}}, index_batch = eachindex(ti, data), params = nn.params) =
mapreduce((args...)->loss_single(ti, nn, get_data(data, :q,args...), get_data(data, :q̇, args...), get_data(data, :q̈, args...), params), +, index_batch)

