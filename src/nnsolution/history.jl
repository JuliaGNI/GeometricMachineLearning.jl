#=
    SingleHistory structure is designed to store the results of a training course. It includes :
        - paramters: the TrainingParameters used for training,
        - datashape: the AbstractDataShape of the data used,
        - size_data: the size of the data used,
        - loss: the loss during training.
=#

struct SingleHistory{TP <: TrainingParameters, TD, TL}
    parameters::TP
    datashape::TD
    size_data::Int
    loss::TL

    SingleHistory(parameters, datashape, size_data, loss) = new{typeof{parameters}, typeof{datashape}, typeof(loss)}(parameters, datashape, size_data, loss)
end


#=
    History structure is designed to store all the past results. It includes :
        - data: a NamedTuple included all the sizemax th last training,
        - last: the last SingleHistory,
        - size: the number of SingleHistory stored,
        - sizemax: the number max of SingleHistory stored begining by the last one.
=#


mutable struct History
    data::Dict{Symbol, SingleHistory}
    last::SingleHistory
    size::Int
    sizemax::Int

    function History(sg::SingleHistory; sizemax = 10)
        history = Dict()
        new(history, sg, 0, sizemax)
    end
end



@inline size(history::History) = history.size
@inline last(history::History) = history.last 
@inline sizemax(history::History) = history.sizemax

function _add(history::History, sg::SingleHistory)
    
    history.size += 1 

    if history.size > history.sizemax
        delete!(history.data, Symbol("training_"*string(history.size - history.sizemax)))
    end
    
    history.data[Symbol("training_"*string(size_history))] = history.last
    history.last = sg
    
end


function _set_sizemax_history(history::History, sizemax::Int)

    @assert sizemax >= 0

    previous_sizemax = history.sizemax
    history.sizemax = sizemax

    for i in (history.size - previous_sizemax + 1):(history.size - history.sizemax + 1)
        delete!(history.data, Symbol("training_"*string(i)))
    end

end