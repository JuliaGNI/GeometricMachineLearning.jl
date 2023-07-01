using GeometricIntegrators

struct UnknownProblem <: AbstractProblem end

struct SingleHistory{TP <: TrainingParameters, TD}
    parameters::TP
    datashape::TD
    size_data::Int
    loss::TL

    SingleHistory(parameters, datashape, size_data, loss) = new{typeof{parameters}, typeof{datashape}, typeof(loss)}(parameters, datashape, size_data, loss)
end

mutable struct History
    data::NamedTuple
    size::Int
    last::SingleHistory

    function History(sg::SingleHistory)
        history = NamedTuple{(Symbol("training_"*string(size_history)),)}(sg)
        new(history, 1, sg)
    end
end

size_history(history) = history.size
last(history) = history.last 

function _add(history::History, sg::SingleHistory)
    history.size += 1 
    history.last = sg
    history.data = merge(history.data, NamedTuple{(Symbol("training_"*string(size_history)),)}(sg))
end

struct NeuralNetSolution{TNN <: AbstractNeuralNetwork, TP <: AbstractProblem, TL}
    nn::TNN
    problem::TP
    tstep::Real
    loss::TL
    history::NamedTuple

    function NeuralNetSolution(nn::AbstractNeuralNetwork, loss::Real, sh::SingleHistory, problem::GeometricProblem = UnknownProblem, tstep::Real = isUnknown(problem) ? nothing : tstep(problem))
        new{typeof(nn), typeof(problem), typeof(loss)}(nn, problem, loss, tstep, History(sh))
    end

    function NeuralNetSolution(nn::AbstractNeuralNetwork, loss::Real, h::History, problem::GeometricProblem = UnknownProblem, tstep::Real = isUnknown(problem) ? nothing : tstep(problem))
        new{typeof(nn), typeof(problem), typeof(loss)}(nn, problem, loss, tstep, h)
    end
end

update_history(nns::NeuralNetSolution, sg::SingleHistory) = _add(nns.history, sg)


@inline nn(nns::NeuralNetSolution) = nns.nn
@inline problem(nns::NeuralNetSolution) = nns.problem
@inline tstep(nns::NeuralNetSolution) = nns.tstep
@inline loss(nns::NeuralNetSolution) = nns.loss
@inline history(nns::NeuralNetSolution) = nns.history.data
@inline size_history(nns::NeuralNetSolution) = nns.history.size






problem(::AbstractTrainingData) = UnknownProblem
tstep(::AbstractTrainingData) = nothing
shape(::AbstractTrainingData) = nothing
size(::AbstractTrainingData) = 0

