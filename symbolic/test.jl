using GeometricMachineLearning
using LinearAlgebra
using Symbolics

using AbstractNeuralNetworks
import AbstractNeuralNetworks: NeuralNetwork

include("utils.jl")



@variables x, y 

nt = (A=x, B=y)

f = nt.A*nt.B*nt.B+nt.A 

fb = build_function(f, nt...)



