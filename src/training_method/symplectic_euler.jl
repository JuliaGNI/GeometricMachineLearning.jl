
abstract type SymplecticEulerIntegrator <: HnnTrainingMethod end

struct SymplecticEulerIntegratorA <: SymplecticEulerIntegrator end
struct SymplecticEulerIntegratorB <: SymplecticEulerIntegrator end

SEuler(;sqdist = sqeuclidean) =  SEulerA(sqdist = sqdist)
SEulerA(;sqdist = sqeuclidean) = TrainingMethod{SymplecticEulerIntegratorA, PhaseSpaceSymbol, TrajectoryData, typeof(sqdist)}(sqdist)
SEulerB(;sqdist = sqeuclidean) = TrainingMethod{SymplecticEulerIntegratorB, PhaseSpaceSymbol, TrajectoryData, typeof(sqdist)}(sqdist)

            
function loss_single(::TrainingMethod{SymplecticEulerIntegratorA}, nn::AbstractNeuralNetwork{<:HamiltonianArchitecture}, qₙ, qₙ₊₁, pₙ, pₙ₊₁, Δt, params = params(nn))
    dH = vectorfield(nn, [qₙ₊₁...,pₙ...], params)
    sqeuclidean(dH[1],(qₙ₊₁-qₙ)/Δt) + sqeuclidean(dH[2],(pₙ₊₁-pₙ)/Δt)
end

function loss_single(::TrainingMethod{SymplecticEulerIntegratorB}, nn::AbstractNeuralNetwork{<:HamiltonianArchitecture}, qₙ, qₙ₊₁, pₙ, pₙ₊₁, Δt, params = params(nn))
    dH = vectorfield(nn, [qₙ...,pₙ₊₁...], params)
    sqeuclidean(dH[1],(qₙ₊₁-qₙ)/Δt) + sqeuclidean(dH[2],(pₙ₊₁-pₙ)/Δt)
end

get_loss(::TrainingMethod{<:SymplecticEulerIntegrator}, ::AbstractNeuralNetwork{<:HamiltonianArchitecture}, data::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}}, args) = (get_data(data,:q, args...), get_data(data,:q, next(args...)...), get_data(data,:p, args...), get_data(data,:p,next(args...)...), get_Δt(data))

loss(ti::TrainingMethod{<:SymplecticEulerIntegrator}, nn::AbstractNeuralNetwork{<:HamiltonianArchitecture}, data::TrainingData{<:DataSymbol{<:PhaseSpaceSymbol}}, index_batch = eachindex(ti, data), params = params(nn)) = 
mapreduce(args->loss_single(Zygote.ignore_derivatives(ti), nn, get_loss(ti, nn, data, args)..., params),+, index_batch)

min_length_batch(::SymplecticEulerIntegrator) = 2