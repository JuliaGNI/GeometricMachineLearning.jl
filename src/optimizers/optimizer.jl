# The optimizers are GeometricOptimizers': `train!` takes its `TrainingOptimizer`, built on the
# parameters of a network with `TrainingOptimizer(params(nn); algorithm, linesearch, retraction)`.
# The types and the retractions are re-exported here, so that `using GeometricMachineLearning`
# alone trains a network. `update!` is not, and this package defines no method on it: the module
# file imports it so that the name is `GeometricOptimizers`' rather than `ProgressMeter`'s, which
# `using ProgressMeter` would otherwise supply, and a caller takes it from `GeometricOptimizers`.
export TrainingOptimizer
export GradientMethod, MomentumMethod, Adam, AdamWithEuclideanDecay
export AdamOptimizerWithDecay, DecayingStatic
export GlobalSection, global_section, apply_section, apply_section!, update_section!
export global_rep
export Geodesic, Cayley
export geodesic, cayley
export retraction
