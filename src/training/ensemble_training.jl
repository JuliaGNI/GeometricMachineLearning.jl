#=
    EnsembleTraining gathers in one structure severals TrainingSet. This structure is mutable so that we can add new TrainingSet easily. 
=#

mutable struct EnsembleTraining{TS <:AbstractArray{TrainingSet}}
    tab::TS
    size::Int
    shared_nn::Bool
    shared_tp::Bool
    shared_data::Bool

    EnsembleTraining() = new{Vector{TrainingSet}}(Vector{TrainingSet}(), 0, false, false, false)

    EnsembleTraining(ts::TrainingSet) = new{typeof([ts])}([ts],1, true, true, true)

    function EnsembleTraining(args::TrainingSet...) 
        et = EnsembleTraining()
        for ts in args
            push!(et,ts)
        end
    end
end

@inline Base.size(et::EnsembleTraining) = et.size

@inline isnnShared(et::EnsembleTraining) = et.shared_nn
@inline isParametersShared(et::EnsembleTraining) = et.shared_tp
@inline isDataShared(et::EnsembleTraining) = et.shared_data

@inline nn(et::EnsembleTraining) = isnnShared(et) ? nn(et[1]) : @error "The NeuralNetwork is not shared for all TrainingSet."
@inline parameters(et::EnsembleTraining) = isParametersShared(et) ? parameters(et[1]) : @error "The TrainingParameters are not shared for all TrainingSet."
@inline data(et::EnsembleTraining) = isDataShared(et) ? data(et[1]) : @error "The Trainingdata are not shared for all TrainingSet."

Base.getindex(et::EnsembleTraining, n::Int) = et.tab[n]
Base.setindex!(et::EnsembleTraining, value::TrainingSet, n::Int) = et.tab[n] = value
Base.iterate(et::EnsembleTraining, state = 1) = state > size(et) ? nothing : (et[state], state+1)

function Base.push!(et::EnsembleTraining, ts::TrainingSet)
    et.size += 1
    isnnShared(et) && nn(et) == nn(ts) ? nothing : et.shared_nn = false
    isParametersShared(et) && parameters(et) == parameters(ts) ? nothing : et.shared_parameters = false
    isDataShared(et) && data(et) == data(ts) ? nothing : et.shared_data= false
    push!(et.tab, ts)
end

function Base.merge!(et₁::EnsembleTraining, et₂::EnsembleTraining)
    et₁.size += et₂.size
    isnnShared(et₁) && isnnShared(et₂) && nn(et₁) == nn(et₂) ? nothing : et₁.shared_nn = false
    isParametersShared(et₁) && isParametersShared(et₂) && parameters(et₁) == parameters(et₂) ? nothing : et₁.shared_parameters = false
    isDataShared(et₁) && isDataShared(et₂) && data(et₁) == data(et₂) ? nothing : et₁.shared_data= false
    for ts in et₂
        push!(et₁.tab, ts)
    end
end