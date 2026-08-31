abstract type VariationalMethod <: LnnTrainingMethod end
struct VariationalMidPointMethod <: VariationalMethod end
struct VariationalTrapezMethod <: VariationalMethod end

function VariaMidPoint(; sqdist = sqeuclidean)
    TrainingMethod{
        VariationalMidPointMethod, PositionSymbol, TrajectoryData, typeof(sqdist)}(sqdist)
end

# discrete langrangian
function discrete_lagrangian(::TrainingMethod{VariationalMidPointMethod},
        nn::NeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, qₙ₊₁, Δt, params = params(nn))
    nn([(qₙ₊₁+qₙ)/2..., (qₙ₊₁-qₙ)/Δt...], params)
end

# gradient of discrete Lagrangian
function DL(ti::TrainingMethod{<:VariationalMethod},
        nn::NeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, qₙ₊₁, Δt, params = params(nn))
    Zygote.gradient((qₙ, qₙ₊₁)->discrete_lagrangian(ti, nn, qₙ, qₙ₊₁, Δt, params), qₙ, qₙ₊₁)
end
function DL₁(ti::TrainingMethod{<:VariationalMethod},
        nn::NeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, qₙ₊₁, Δt, params = params(nn))
    DL(ti, nn, qₙ, qₙ₊₁, Δt, params)[1:length(qₙ)]
end
function DL₂(ti::TrainingMethod{<:VariationalMethod},
        nn::NeuralNetwork{<:LagrangianNeuralNetwork}, qₙ, qₙ₊₁, Δt, params = params(nn))
    DL(ti, nn, qₙ, qₙ₊₁, Δt, params)[(1 + length(qₙ)):end]
end

function loss_single(ti::TrainingMethod{<:VariationalMethod},
        nn::AbstractNeuralNetwork{<:LagrangianNeuralNetwork},
        qₙ, qₙ₊₁, qₙ₊₂, Δt, params = params(nn))
    DL1 = DL₁(ti, nn, qₙ₊₁, qₙ₊₂, Δt, params)
    DL2 = DL₂(ti, nn, qₙ, qₙ₊₁, Δt, params)
    sqeuclidean(DL1, -DL2)
end

function get_loss(::TrainingMethod{<:VariationalMidPointMethod},
        ::AbstractNeuralNetwork{<:LagrangianNeuralNetwork},
        data::TrainingData{<:DataSymbol{<:PositionSymbol}}, args)
    (get_data(data, :q, args...), get_data(data, :q, next(args...)...),
        get_data(data, :q, next(next(args...)...)...), get_Δt(data))
end

function loss(ti::TrainingMethod{<:VariationalMidPointMethod},
        nn::AbstractNeuralNetwork{<:LagrangianNeuralNetwork},
        data::TrainingData{<:DataSymbol{<:PositionSymbol}},
        index_batch = eachindex(ti, data), params = params(nn))
    mapreduce(
        args->loss_single(Zygote.ignore_derivatives(ti), nn, get_loss(ti, nn, data, args)..., params),
        +,
        index_batch)
end
min_length_batch(::VariationalMethod) = 3
