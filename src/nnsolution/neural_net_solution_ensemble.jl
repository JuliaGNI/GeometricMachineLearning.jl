#=
    EnsembleNeuralNetSolution gathers severals NeuralNetSolution. It is the results given by a training of an EnsembleTraining.
=#

mutable struct EnsembleNeuralNetSolution{TNNS <: AbstractArray{<:NeuralNetSolution}}
    tab::TNNS
    size::Int

    EnsembleNeuralNetSolution() = new{Vector{NeuralNetSolution}}(Vector{NeuralNetSolution}(), 0)

    EnsembleNeuralNetSolution(nns::NeuralNetSolution) = new{typeof([nns])}([nns],1)

    function EnsembleNeuralNetSolution(args::NeuralNetSolution...) 
        enns = EnsembleNeuralNetSolution()
        for nns in args
            push!(enns,nns)
        end
    end
end

@inline Base.size(enns::EnsembleNeuralNetSolution) = enns.size

Base.getindex(enns::EnsembleNeuralNetSolution, n::Int) = enns.tab[n]
Base.setindex!(enns::EnsembleNeuralNetSolution, value::NeuralNetSolution, n::Int) = enns.tab[n] = value
Base.iterate(enns::EnsembleNeuralNetSolution, state = 1) = state > size(enns) ? nothing : (enns[state], state+1)

function Base.push!(enns::EnsembleNeuralNetSolution, nns::NeuralNetSolution)
    enns.size += 1
    push!(enns.tab, nns)
end

function Base.merge!(enns₁::EnsembleNeuralNetSolution, enns₂::EnsembleNeuralNetSolution)
    enns₁.size += enns₂.size
    for nns in enns₂
        push!(enns₁.tab, nns)
    end
end