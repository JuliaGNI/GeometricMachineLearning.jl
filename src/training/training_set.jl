#=
    SingleTrainingSet gathers all information ready for a train. It contains
        - nn : an AbstractNeuralNetwork
        - tp : a TrainingParameters
        - data : an AbstractTrainingData
=#

struct SingleTrainingSet{TN <: AbstractNeuralNetwork, TP<:TrainingParameters , TD<: AbstractTrainingData}
    nn::TN
    tp::TP
    data::TD

    function SingleTrainingSet(nn::AbstractNeuralNetwork, tp::TrainingParameters, data::AbstractTrainingData)
        new{typeof(nn), typeof(tp), typeof(dada)}(nn,tp,data)
    end
end

SingleTrainingSet(nn::AbstractNeuralNetwork, tp::TrainingParameters, datashape::NamedTuple, problem::AbstractProblem, data) = SingleTrainingSet(nn, tp, TrainingData(data, datashape, problem))

SingleTrainingSet(sts::SingleTrainingSet; nn::AbstractNeuralNetwork = nn(sts), tp::TrainingParameters = parameters(sts), data::AbstractTrainingData = data(sts)) = SingleTrainingSet(nn, tp, data)

struct TrainingSets{TS <:AbstractArray{SingleTrainingSet}}
    tab::TS
    size::Int
    shared_nn::Bool
    shared_tp::Bool
    shared_data::Bool
end


@inline nn(sts::SingleTrainingSet)= sts.nn
@inline parameters(sts::SingleTrainingSet)= sts.tp
@inline data(sts::SingleTrainingSet)= sts.data
@inline size(ts::TrainingSets) = ts.size


Base.getindex(ts::TrainingSets, n::Int) = ts.tab[n]
Base.setindex!(ts::TrainingSets, value::SingleTrainingSet, n::Int) = ts.tab[n] = value

