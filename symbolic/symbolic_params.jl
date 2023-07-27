using GeometricMachineLearning
using LinearAlgebra
using Symbolics

using AbstractNeuralNetworks
import AbstractNeuralNetworks: NeuralNetwork

include("utils.jl")


function symbolicParams(M::AbstractArray, i::Int = 0)
    sname = SymbolicName(Symbol(@Name M),i)
    ((@variables $sname[Tuple([1:s for s in size(M)])...])[1],i+1)
end

function symbolicParams(nt::NamedTuple, i::Int = 0)
    if length(nt) == 1
        symb, j = symbolicParams(values(nt)[1], i)
        return NamedTuple{keys(nt)}((symb,)),j
    else
        symb, j = symbolicParams(values(nt)[1], i)
        symbs, k = symbolicParams(NamedTuple{keys(nt)[2:end]}(values(nt)[2:end]), j)
        return  (NamedTuple{keys(nt)}(Tuple([symb, symbs...])), k)
    end

end

function symbolicParams(t::Tuple, i::Int = 0)
    if length(t) == 1
        symb, j = symbolicParams(t[1],i)
        return (symb,),j
    else
        symb, j = symbolicParams(t[1],i)
        symbs, k = symbolicParams(t[2:end], j)
        return (Tuple([symb, symbs...]),k)
    end
end

symbolicParams(nn::NeuralNetwork) = symbolicParams(nn.params,1)[1]