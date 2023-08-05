struct LnnExactIntegrator <: LnnTrainingIntegrator end

ExactLnn(;sqdist = sqeuclidean) = TrainingIntegrator{LnnExactIntegrator, PosVeloAccSymbol, SampledData, typeof(sqdist)}(sqdist)

function loss_single(::TrainingIntegrator{LnnExactIntegrator}, nn::NeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, q̇ₙ, q̈ₙ, params = nn.params)
    abs(sum(∇q∇q̇L(nn,qₙ, q̇ₙ, params)))  #inv(∇q̇∇q̇L(nn, qₙ, q̇ₙ, params))*(∇qL(nn, qₙ, q̇ₙ, params) - ∇q∇q̇L(nn, qₙ, q̇ₙ, params))
end

loss(ti::TrainingIntegrator{<:LnnExactIntegrator}, nn::AbstractNeuralNetwork{<:LagrangianNeuralNetwork}, data::TrainingData{<:DataSymbol{<:PosVeloAccSymbol}}, index_batch = eachindex(ti, data), params = nn.params) = 
mapreduce(args->loss_single(Zygote.ignore(ti), nn, get_loss(ti, nn, data, args)..., params),+, index_batch)

get_loss(::TrainingIntegrator{<:LnnExactIntegrator}, ::AbstractNeuralNetwork{<:LagrangianNeuralNetwork}, data::TrainingData{<:DataSymbol{<:PosVeloAccSymbol}}, args) = (get_data(data, :q,args...), get_data(data, :q̇, args...), get_data(data, :q̈, args...))