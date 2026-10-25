module GeometricMachineLearning

using AbstractNeuralNetworks
# The parameter container lives in `NeuralNetworkParameters` as of `AbstractNeuralNetworks` 0.7,
# under the name `NetworkParameters`. The import is selective rather than a bare `using`: that
# package also exports `flatten`/`unflatten` and the leaf protocol, none of which this package
# extends — `GeometricOptimizers` carries the protocol for the structured matrices.
#
# `mapstorage` comes from there rather than being written again here: it reaches the storage of a
# structured leaf and rebuilds the leaf around the result (`src/map_to_cpu.jl`). A plain
# `NamedTuple` of layers is walked with `Base.map`, which needs nothing from anybody.
import NeuralNetworkParameters: NetworkParameters
using NeuralNetworkParameters: mapstorage
using ChainRulesCore
using GeometricBase
using GeometricSolutions: GeometricSolution, EnsembleSolution, TimeSeries
using GeometricEquations: HODEProblem, HODEEnsemble
using KernelAbstractions
using LinearAlgebra
using NNlib
using ProgressMeter
using Random
using Zygote
using ForwardDiff
import DiffResults
using InteractiveUtils
using TimerOutputs

# The manifolds, the structured matrix types, the global sections and the retractions are
# `GeometricOptimizers`': the same objects, so upstream's generic machinery dispatches on them and
# the retraction pipeline is written once, there.
#
# `import` and not `using ...: ...`: GML adds constructor methods to several of these types (in
# `layers/` and the kernels), and extending a *type* reached through `using` warns on every such
# method since Julia 1.12 — "Constructor for type … was extended in `GeometricMachineLearning`
# without explicit qualification or import". Everything imported here is re-exported below, except
# `orthonormal_columns` and `optimization_step!`; the re-exports are what keep
# `using GeometricMachineLearning` alone sufficient.
import GeometricOptimizers
import GeometricOptimizers: Manifold, StiefelManifold, GrassmannManifold
import GeometricOptimizers: SkewSymMatrix, SymmetricMatrix,
                            StrictlyLowerTriangular, StrictlyUpperTriangular,
                            StiefelProjection
import GeometricOptimizers: StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix
import GeometricOptimizers: rgrad, metric, check, global_section
# `orthonormal_columns(draw)` orthonormalizes `draw()` with CholeskyQR2, which runs on whatever
# backend `draw` allocated on. The three manifold layers below initialise their weights with it,
# because `LinearAlgebra.qr!` is a host factorization and `Metal` implements no `qr` at all.
import GeometricOptimizers: orthonormal_columns
import GeometricOptimizers: GlobalSection, global_rep, apply_section, apply_section!,
                            update_section!
import GeometricOptimizers: Geodesic, Cayley, geodesic, cayley, retraction
import GeometricOptimizers: GradientMethod, MomentumMethod, Adam, AdamWithEuclideanDecay,
                            AdamOptimizerWithDecay, DecayingStatic
# The training step. `train!` takes a `TrainingOptimizer` and calls `optimization_step!`, which is
# not re-exported.
import GeometricOptimizers: TrainingOptimizer, optimization_step!
import GeometricOptimizers: update!
# `solve!` is imported rather than started afresh so that GML's `solve!(::NeuralNetwork{<:PSDArch},
# …)` — solve for the parameters directly, by SVD, instead of training for them — is a method of the
# same verb a caller already has from GeometricOptimizers, and not a second function of the name.
import GeometricOptimizers: solve!

import AbstractNeuralNetworks: Architecture, AbstractExplicitLayer, NeuralNetwork,
                               FeedForwardLoss
import AbstractNeuralNetworks: Chain
import AbstractNeuralNetworks: input_dimension, output_dimension
import AbstractNeuralNetworks: Dense, Linear
# `update!` is deliberately not among these: `AbstractNeuralNetworks`' is a different generic
# function from `GeometricOptimizers.update!` — which is `GeometricBase.update!`, the one with
# methods for the optimizer caches. GML imports that one, from GeometricOptimizers, above.
import AbstractNeuralNetworks: initialparameters
import AbstractNeuralNetworks: parameterlength
import AbstractNeuralNetworks: GlorotUniform
import AbstractNeuralNetworks: params, model, dim
import AbstractNeuralNetworks: AbstractPullback, NetworkLoss, _compute_loss
import AbstractNeuralNetworks: networkbackend
# The input and output types of a network: `NamedTupleOfArrays{T, (:q, :p)}` is data in `(q, p)`
# form, and `ArrayOrNamedTuple{T}` is an array or a named tuple of arrays with any keys.
using AbstractNeuralNetworks: NamedTupleOfArrays, ArrayOrNamedTuple
# `save` and `load` are `NeuralNetworkParameters`' generics; `AbstractNeuralNetworks` 0.7 only
# re-binds them. Reach for them where they are defined.
import NeuralNetworkParameters: save, load
export dim
import NNlib: σ, sigmoid, softmax
import Base: iterate, eltype

export CPU, GPU
export Chain, NeuralNetwork
export Dense, Linear
export initialparameters
export parameterlength
export NetworkParameters

export σ, sigmoid, softmax

# `GeometricBase` defines `description` but does not export it, so `using GeometricBase` alone does
# not bring it into scope and re-exporting it needs the explicit import. `StateVariable` is reached
# from here rather than from `GeometricSolutions`, which only re-exports it: importing a name from a
# module that does not own it is what `check_all_explicit_imports_via_owners` reports.
import GeometricBase: description, StateVariable
export description

include("utils.jl")

include("data_loader/data_loader.jl")

# INCLUDE ARRAYS — the structured matrix types come from GeometricOptimizers, which also carries
# every method GML needs on them. `PoissonTensor` is GML's own and is the only one defined here.
include("arrays/poisson_tensor.jl")

# Re-exported from GeometricOptimizers, so that `using GeometricMachineLearning` on its own still
# gives a caller the matrix types its layers are parametrized by.
export SymmetricMatrix, SkewSymMatrix
export StrictlyLowerTriangular, StrictlyUpperTriangular
export StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix
export StiefelProjection
# GML's own
export PoissonTensor

include("kernels/assign_q_and_p.jl")
include("kernels/tensor_tensor_mul.jl")
include("kernels/tensor_transpose_tensor_mul.jl")
include("kernels/tensor_tensor_transpose_mul.jl")
include("kernels/mul.jl")
include("kernels/cayley.jl")
include("kernels/inverses/tensor_mat_skew_sym_assign.jl")

include("kernels/kernel_ad_routines/seed.jl")
include("kernels/kernel_ad_routines/tensor_tensor_mul.jl")
include("kernels/kernel_ad_routines/tensor_transpose_tensor_mul.jl")
include("kernels/kernel_ad_routines/tensor_mat_skew_sym_assign.jl")

export MatrixSoftmax, VectorSoftmax
include("activations/softmax.jl")

# `_diff` and `_norm` are the `NamedTuple`/`(q, p)` arms of subtraction and the norm, and neither is
# exported: they are helpers of `src/reduced_system/`, not surface. Nothing named `_add` stands
# beside them, and nothing here adds a method to `AbstractNeuralNetworks.add!`: that would be
# piracy, and `GeometricOptimizers` already defines the three-argument `add!` on the structured
# matrix types against its own generic, for a caller that wants them.

export GradientLayerQ, GradientLayerP, ActivationLayerQ, ActivationLayerP, LinearLayerQ,
       LinearLayerP
export Linear

# The manifolds are GeometricOptimizers' too, along with the geometry that goes with them.
export StiefelManifold, GrassmannManifold, Manifold
export rgrad, metric, check

include("layers/sympnets.jl")
include("layers/bias_layer.jl")
include("layers/resnet.jl")
include("layers/manifold_layer.jl")
include("layers/stiefel_layer.jl")
include("layers/grassmann_layer.jl")
include("layers/positional_encoding.jl")
include("layers/multi_head_attention.jl")
include("layers/volume_preserving_attention.jl")
include("layers/volume_preserving_feedforward.jl")
include("layers/transformer.jl")
include("layers/psd_like_layer.jl")
include("layers/classification.jl")

export StiefelLayer, GrassmannLayer, ManifoldLayer
export PSDLayer
export MultiHeadAttention
export PositionalEncoding, positional_encoding
export VolumePreservingAttention
export VolumePreservingFeedForwardLayer
export VolumePreservingLowerLayer
export VolumePreservingUpperLayer
export VolumePreservingTransformer
export NeuralNetworkIntegrator
export ResNet
export Transformer
export TransformerIntegrator, StandardTransformerIntegrator

# INCLUDE OPTIMIZERS — the methods, sections and retractions come from GeometricOptimizers.
include("optimizers/optimizer.jl")

export NeuralNetwork

export NetworkLoss, TransformerLoss, FeedForwardLoss, AutoEncoderLoss, ReducedLoss

#INCLUDE ARCHITECTURES
include("architectures/neural_network_integrator.jl")
include("architectures/resnet.jl")
include("architectures/transformer_integrator.jl")
include("architectures/standard_transformer_integrator.jl")
include("architectures/sympnet.jl")
include("architectures/autoencoder.jl")
include("architectures/symplectic_autoencoder.jl")
include("architectures/psd.jl")
include("architectures/transformer_neural_network.jl")
include("architectures/volume_preserving_feedforward.jl")
include("architectures/volume_preserving_transformer.jl")

export SympNet, LASympNet, GSympNet
export ClassificationTransformer, ClassificationLayer
export VolumePreservingFeedForward
export SymplecticAutoencoder, PSDArch

export solve!, encoder, decoder

export iterate

include("loss/losses.jl")

export AbstractPullback, ZygotePullback
include("pullbacks/zygote_pullback.jl")

export DataLoader
export Batch, train!
include("data_loader/tensor_assign.jl")
include("data_loader/batch.jl")
include("data_loader/train.jl")

include("reduced_system/reduced_system.jl")

export HRedSys, reduction_error, projection_error, integrate_reduced_system,
       integrate_full_system

include("layers/linear_symplectic_attention.jl")
include("layers/symplectic_attention.jl")
include("architectures/linear_symplectic_transformer.jl")
include("architectures/symplectic_transformer.jl")

export LinearSymplecticAttention, LinearSymplecticAttentionQ, LinearSymplecticAttentionP
export LinearSymplecticTransformer
export SymplecticAttention, SymplecticAttentionQ, SymplecticAttentionP
export SymplecticTransformer

include("map_to_cpu.jl")

export save, load
end
