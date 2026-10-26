# The optimizers are GeometricOptimizers': `train!` takes its `TrainingOptimizer`, built on the
# parameters of a network with `TrainingOptimizer(params(nn); algorithm, linesearch, retraction)`.
# The types and the retractions are re-exported here, so that `using GeometricMachineLearning`
# alone trains a network. `update!` is not: it is `GeometricOptimizers`' verb, this package only
# adds methods to it, and a caller who wants it takes it from `GeometricOptimizers`.
export TrainingOptimizer
export GradientMethod, MomentumMethod, Adam, AdamWithEuclideanDecay
export AdamOptimizerWithDecay, DecayingStatic
export GlobalSection, global_section, apply_section, apply_section!, update_section!
export global_rep
export Geodesic, Cayley
export geodesic, cayley
export retraction
