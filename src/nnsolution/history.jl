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

function show(sh::SingleHistory, head = true)
    if head

    end
    
end


#=
    History structure is designed to store all the past results. It includes :
        - data: a NamedTuple included all the sizemax th last training,
        - last: the last SingleHistory,
        - size: the number of SingleHistory stored,
        - sizemax: the number max of SingleHistory stored begining by the last one.
=#


mutable struct History
    data::AbstractArray{SingleHistory}
    last::SingleHistory
    size::Int
    sizemax::Int
    nbtraining::Int

    function History(sg::SingleHistory; sizemax = 100)
        history = Dict()
        new(history, sg, 0, sizemax)
    end
end

@inline data(history::History) = history.data
@inline last(history::History) = history.last 
@inline size(history::History) = history.size
@inline sizemax(history::History) = history.sizemax
@inline nbtraining(history::History) = history.nbtraining

Base.getindex(history::History, n::Int) = data(history)[n]
Base.iterate(history::History, state = 1) = state > size(history) ? nothing : (history[state],state+1)

function _add(history::History, sg::SingleHistory)
    
    history.nbtraining += 1 

    history.size == history.sizemax ? popfirst!(history.data) : history.size += 1

    push!(history.data, history.last)
        
    history.last = sg
    
end


function _set_sizemax_history(history::History, sizemax::Int)

    @assert sizemax >= 0

    history.sizemax = sizemax

    for _ in 1:(size(history)-sizemax(history))
        popfirst!(history.data)
    end

    history.size = history.size - max(history.size - history.sizemax, 0)

end


function show(history::History)
    println("Print of history : ")
    println("-------------------")
    println("Last training :", )

end