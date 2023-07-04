struct HnnExactIntegrator <: HnnTrainingIntegrator end

ExactHnn(;sqdist = sqeuclidean) = TrainingIntegrator(HnnExactIntegrator(); sqdist = sqdist)

function loss_single(::HnnExactIntegrator, nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, qₙ, pₙ, q̇ₙ, ṗₙ, params = nn.params)
    dH = vectorfield(nn, [qₙ...,pₙ...], params)
    sqeuclidean(dH[1],q̇ₙ) + sqeuclidean(dH[2],ṗₙ)
end


#loss(ti::HnnExactIntegrator, nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, datat::TrainingData{DataSymbol{<:PhaseSpaceSymbol}}, index_batch = get_batch(datat), params = nn.params) =
#mapreduce(x->loss_single(ti, nn, get_data(datat, :q,x[1],x[2]), get_data(datat,:p,x[1],x[2]), get_target(datat,:q̇,x[1],x[2]), get_target(datat,:ṗ,x[1],x[2]), params), +, index_batch)
    

loss(ti::HnnExactIntegrator, nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, datat::TrainingData{<:DataSymbol{<:DerivativePhaseSpaceSymbol}}, index_batch = get_batch(datat), params = nn.params) = 
mapreduce((args...)->loss_single(ti, nn, get_data(datat,:q,args...), get_data(datat,:p,args...), get_target(datat,:q̇,args...), get_target(datat,:ṗ,args...), params), +, index_batch)


required_key(::HnnExactIntegrator) = (:q,:p, :q̇, :ṗ)

data_goal(::HnnExactIntegrator) = (test_data_target,)