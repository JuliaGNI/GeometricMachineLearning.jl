#=
    NeuralNetSolution structure is designed to store the results of a training on a neural network. It includes :
        - nn: the NeuralNetwork trained,
        - problem: the AbstractProblem on which was trained the NeuralNetwork if it is known (else it is UnknownProblem),
        - timestep: the time step of between consecutive data points if there is one (nothing else),
        - loss: a vector containing the loss during the training,
        - history: an History which contains the last training on the NeuralNetwork.

    A NeuralNetSolution can so be put as argument of train! to perform a new training on the NeuralNetwork.
=#

struct NeuralNetSolution{TNN <: NeuralNetwork, TP <: AbstractProblem, timestep <: Union{Nothing, Real}, TL}
    nn::TNN
    problem::TP
    timestep::timestep
    loss::TL
    history::History

    function NeuralNetSolution(nn::NeuralNetwork, sh::SingleHistory, loss, problem::AbstractProblem = UnknownProblem, timestep::Real = isUnknown(problem) ? nothing : timestep(problem))
        new{typeof(nn), typeof(problem), typeof(timestep), typeof(loss)}(nn, problem, timestep, loss, History(sh))
    end

    function NeuralNetSolution(nn::NeuralNetwork, h::History, loss, problem::AbstractProblem = UnknownProblem, timestep::Real = isUnknown(problem) ? nothing : timestep(problem))
        new{typeof(nn), typeof(problem), typeof(timestep), typeof(loss)}(nn, problem, timestep, loss, h)
    end
end

update_history(nns::NeuralNetSolution, sg::SingleHistory) = _add(nns.history, sg)

@inline nn(nns::NeuralNetSolution) = nns.nn
@inline problem(nns::NeuralNetSolution) = nns.problem
@inline problem(nns::NeuralNetSolution{T,<:EnsembleProblem} where T) = GeometricEquations.problem(nns.problem,1)
@inline GeometricBase.timestep(nns::NeuralNetSolution) = nns.timestep
@inline loss(nns::NeuralNetSolution) = nns.loss

@inline history(nns::NeuralNetSolution) = nns.history
@inline size_history(nns::NeuralNetSolution) = size(nns.history)
@inline Base.last(nns::NeuralNetSolution) = last(nns.history)
@inline nbtraining(nns::NeuralNetSolution) =  nbtraining(nns.history)

set_sizemax_history(nns::NeuralNetSolution, sizemax::Int) = _set_sizemax_history(nns.history, sizemax)

show_history(nns::NeuralNetSolution) = show(history(nns))
