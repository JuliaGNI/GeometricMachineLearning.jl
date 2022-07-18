using GeometricMachineLearning
using Test

w = rand(2,2)

sq = LinearSymplecticLayerQ(w, x -> 1)
sp = LinearSymplecticLayerP(w, x -> 1)

@test typeof(sq) <: LinearFeedForwardLayer
@test typeof(sp) <: LinearFeedForwardLayer

@test typeof(sq) <: LinearSymplecticLayerQ
@test typeof(sp) <: LinearSymplecticLayerP
