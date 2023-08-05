
abstract type SymplecticEuler <: HnnTrainingMethod end

struct SymplecticEulerA <: SymplecticEuler end
struct SymplecticEulerB <: SymplecticEuler end

SEuler(;sqdist = sqeuclidean) =  SEulerA(sqdist = sqdist)
SEulerA(;sqdist = sqeuclidean) = TrainingMethod{SymplecticEulerA, PhaseSpaceSymbol, TrajectoryData, typeof(sqdist)}(sqdist)
SEulerB(;sqdist = sqeuclidean) = TrainingMethod{SymplecticEulerB, PhaseSpaceSymbol, TrajectoryData, typeof(sqdist)}(sqdist)

            
function loss_single(::TrainingMethod{SymplecticEulerA}, nn::NeuralNetwork{<:HamiltonianNeuralNetwork}, qₙ, qₙ₊₁, pₙ, pₙ₊₁, Δt, params = nn.params)
    dH = vectorfield(nn, [qₙ₊₁...,pₙ...], params)
    sqeuclidean(dH[1],(qₙ₊₁-qₙ)/Δt) + sqeuclidean(dH[2],(pₙ₊₁-pₙ)/Δt)
end

function loss_single(::TrainingMethod{SymplecticEulerB}, nn::NeuralNetwork{<:HamiltonianNeuralNetwork}, qₙ, qₙ₊₁, pₙ, pₙ₊₁, Δt, params = nn.params)
    dH = vectorfield(nn, [qₙ...,pₙ₊₁...], params)
    sqeuclidean(dH[1],(qₙ₊₁-qₙ)/Δt) + sqeuclidean(dH[2],(pₙ₊₁-pₙ)/Δt)
end

get_loss(::TrainingMethod{<:SymplecticEuler}, ::AbstractNeuralNetwork{<:HamiltonianNeuralNetwork}, data::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}}, args) = (get_data(data,:q, args...), get_data(data,:q, next(args...)...), get_data(data,:p, args...), get_data(data,:p,next(args...)...), get_Δt(data))

loss(ti::TrainingMethod{<:SymplecticEuler}, nn::AbstractNeuralNetwork{<:HamiltonianNeuralNetwork}, data::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}}, index_batch = eachindex(ti, data), params = nn.params) = 
mapreduce(args->loss_single(Zygote.ignore(ti), nn, get_loss(ti, nn, data, args)..., params),+, index_batch)

min_length_batch(::SymplecticEuler) = 2

