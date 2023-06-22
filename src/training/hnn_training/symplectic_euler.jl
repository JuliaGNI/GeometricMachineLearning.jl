
abstract type SymplecticEuler <: HnnTrainingIntegrator end

struct SymplecticEulerA <: SymplecticEuler end
struct SymplecticEulerB <: SymplecticEuler end

#SymplecticEulerA(;sqdist = sqeuclidean) = TrainingIntegrator(_SymplecticEulerA(), sqdist)
#SymplecticEulerB(;sqdist = sqeuclidean) = TrainingIntegrator(_SymplecticEulerB(), sqdist)

            
function loss_single(::SymplecticEulerA, nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, qₙ, qₙ₊₁, pₙ, pₙ₊₁, Δt, params = nn.params)
    dH = vectorfield(nn, [qₙ₊₁...,pₙ...], params)
    sqeuclidean(dH[1],(qₙ₊₁-qₙ)/Δt) + sqeuclidean(dH[2],(pₙ₊₁-pₙ)/Δt)
end

function loss_single(::SymplecticEulerB, nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, qₙ, qₙ₊₁, pₙ, pₙ₊₁, Δt, params = nn.params)
    dH = vectorfield(nn, [qₙ...,pₙ₊₁...], params)
    sqeuclidean(dH[1],(qₙ₊₁-qₙ)/Δt) + sqeuclidean(dH[2],(pₙ₊₁-pₙ)/Δt)
end


loss(ti::SymplecticEuler, nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, data::DataTrajectory, index_batch = get_batch(data), params = nn.params) = 
mapreduce(x->loss_single(Zygote.ignore(ti), nn, data.get_data[:q](x[1],x[2]), data.get_data[:q](x[1],x[2]+1), data.get_data[:p](x[1],x[2]), data.get_data[:p](x[1],x[2]+1), data.get_Δt(), params),+, index_batch)

data_goal(::SymplecticEuler) = DataTrajectory

required_key(::SymplecticEuler) = (:q,:p)