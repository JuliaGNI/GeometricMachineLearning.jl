struct HnnExactIntegrator <: HnnTrainingIntegrator end


function loss_single(::HnnExactIntegrator, nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, qₙ, pₙ, q̇ₙ, ṗₙ, params = nn.params)
    dH = vectorfield(nn, [qₙ...,pₙ...], params)
    sqeuclidean(dH[1],q̇ₙ) + sqeuclidean(dH[2],ṗₙ)
end


loss(ti::HnnExactIntegrator, nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, datat::DataTarget{DataTrajectory}, index_batch = get_batch(datat), params = nn.params) =
mapreduce(x->loss_single(ti, nn, datat.get_data[:q](x[1],x[2]), datat.get_data[:p](x[1],x[2]), datat.get_target[:q̇](x[1],x[2]), datat.get_target[:ṗ](x[1],x[2]), params), +, index_batch)
    

loss(ti::HnnExactIntegrator, nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, datat::DataTarget{DataSampled}, index_batch = get_batch(datat), params = nn.params) = 
mapreduce(n->loss_single(ti, nn, datat.get_data[:q](n), datat.get_data[:p](n), datat.get_target[:q̇](n), datat.get_target[:ṗ](n), params), +, index_batch)


required_key(::HnnExactIntegrator) = (:q,:p, :q̇, :ṗ)

