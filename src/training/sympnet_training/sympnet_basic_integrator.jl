struct BasicSympNetIntegrator <: SympNetTrainingIntegrator end

BasicSympNet(;sqdist = sqeuclidean) = TrainingIntegrator(BasicSympNetIntegrator(); sqdist = sqdist)

function loss_single(::BasicSympNetIntegrator, nn::LuxNeuralNetwork{<:SympNet}, qₙ, pₙ, qₙ₊₁, pₙ₊₁, params = nn.params)
    q̃ₙ₊₁,p̃ₙ₊₁ = nn([qₙ...,pₙ...],params)
    sqeuclidean(q̃ₙ₊₁,qₙ₊₁) + sqeuclidean(p̃ₙ₊₁,pₙ₊₁)
end

loss(ti::BasicSympNetIntegrator, nn::LuxNeuralNetwork{<:SympNet}, datat::DataTrajectory, index_batch = get_batch(datat), params = nn.params) =
mapreduce(x->loss_single(ti, nn, Zygote.ignore(get_data(datat,:q,x[1],x[2])), Zygote.ignore(get_data(datat,:p,x[1],x[2])), Zygote.ignore(get_data(datat,:q,x[1],x[2]+1)), Zygote.ignore(get_data(datat,:p,x[1],x[2]+1)), params), +, index_batch)

required_key(::BasicSympNetIntegrator) = (:q,:p)

min_length_batch(::BasicSympNetIntegrator) = 3

data_goal(::BasicSympNetIntegrator) = (test_data_trajectory,)