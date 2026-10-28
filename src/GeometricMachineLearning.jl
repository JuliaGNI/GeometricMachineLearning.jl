module GeometricMachineLearning

using AbstractNeuralNetworks
# `NetworkParameters` is `NeuralNetworkParameters`' parameter container, and `mapstorage` its walk
# of a structured leaf's storage, which rebuilds the leaf around the result (`src/map_to_cpu.jl`).
# The import is selective rather than a bare `using`: that package also exports `flatten`/
# `unflatten` and the leaf protocol, none of which this package extends — `GeometricOptimizers`
# carries the protocol for the structured matrices.
import NeuralNetworkParameters: NetworkParameters
using NeuralNetworkParameters: mapstorage
# `flatten` puts a system's parameters into the input of a network that takes them
# (`src/parametric.jl`); `parameter_eltype` and `parameterlayout` check their element type and
# shape, and `map_cotangent` and `storage_gradient` turn a cotangent into a gradient.
using NeuralNetworkParameters: flatten, parameter_eltype, parameterlayout, map_cotangent,
                               storage_gradient
using ChainRulesCore
using GeometricBase
# `integrate` is GeometricBase's generic, and importing it is what makes the trajectory methods
# below methods of that function rather than of a second one of the name.
import GeometricBase: integrate
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

# The manifolds, the structured matrix types, the global sections and the retractions are
# `GeometricOptimizers`': the same objects, so upstream's machinery dispatches on them and the
# retraction pipeline is written once, there.
#
# `import` and not `using ...: ...`, because a constructor method added to a *type* reached
# through `using` warns on every such method — "Constructor for type … was extended in
# `GeometricMachineLearning` without explicit qualification or import". The re-exports below are
# what keep `using GeometricMachineLearning` alone sufficient. Of these imports, `update!`,
# `solve!`, `optimization_step!` and `orthonormal_columns` are not re-exported: a caller reaches
# `update!` and `solve!` from `GeometricOptimizers`, which exports both, and the other two are
# internal.
import GeometricOptimizers
import GeometricOptimizers: Manifold, StiefelManifold, GrassmannManifold
import GeometricOptimizers: SkewSymMatrix, SymmetricMatrix,
                            StrictlyLowerTriangular, StrictlyUpperTriangular,
                            StiefelProjection
import GeometricOptimizers: StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix
import GeometricOptimizers: rgrad, metric, check, global_section
# `orthonormal_columns(draw)` orthonormalizes `draw()` with CholeskyQR2 on `draw`'s own backend; the
# manifold layers draw through it, because `LinearAlgebra.qr!` is host-only and `Metal` has no `qr`.
import GeometricOptimizers: orthonormal_columns
import GeometricOptimizers: GlobalSection, global_rep, apply_section, apply_section!,
                            update_section!
import GeometricOptimizers: Geodesic, Cayley, geodesic, cayley, retraction
import GeometricOptimizers: GradientMethod, MomentumMethod, Adam, AdamWithEuclideanDecay,
                            AdamOptimizerWithDecay, DecayingStatic
# The training step: `train!` takes a `TrainingOptimizer` and calls `optimization_step!`.
import GeometricOptimizers: TrainingOptimizer, optimization_step!
import GeometricOptimizers: update!
# `solve!` is imported rather than started afresh, so that GML's `solve!` on a `PSDArch` network is
# a method of the verb a caller already has from GeometricOptimizers, not a second one of the name.
import GeometricOptimizers: solve!

import AbstractNeuralNetworks: Architecture, AbstractExplicitLayer, NeuralNetwork,
                               FeedForwardLoss
import AbstractNeuralNetworks: Chain
import AbstractNeuralNetworks: input_dimension, output_dimension
import AbstractNeuralNetworks: Dense, Linear
# `update!` is not among these: `AbstractNeuralNetworks.update!` is a different generic from
# `GeometricOptimizers.update!`, which is `GeometricBase.update!`, the one with the cache methods.
import AbstractNeuralNetworks: initialparameters
import AbstractNeuralNetworks: parameterlength
import AbstractNeuralNetworks: GlorotUniform
import AbstractNeuralNetworks: params, model, dim
import AbstractNeuralNetworks: AbstractPullback, NetworkLoss, _compute_loss
import AbstractNeuralNetworks: networkbackend
# The input and output types of a network: `NamedTupleOfArrays{T, (:q, :p)}` is data in `(q, p)`
# form, and `ArrayOrNamedTuple{T}` is an array or a named tuple of arrays with any keys.
using AbstractNeuralNetworks: NamedTupleOfArrays, ArrayOrNamedTuple
# `save` and `load` are `NeuralNetworkParameters`' generics, which `AbstractNeuralNetworks`
# re-binds; reach for them where they are defined.
import NeuralNetworkParameters: save, load
export dim
import NNlib: σ, sigmoid, softmax
import Base: eltype

export CPU, GPU
export Chain, NeuralNetwork
export Dense, Linear
export initialparameters
export parameterlength
export NetworkParameters

export σ, sigmoid, softmax

# `GeometricBase` defines `description` and does not export it, so re-exporting it needs the import.
# `StateVariable` is taken from here, its owner, and not from `GeometricSolutions`, which only
# re-binds it: importing a name from a non-owner is what `check_all_explicit_imports_via_owners`
# reports.
import GeometricBase: description, StateVariable
export description

include("utils.jl")

include("data_loader/data_loader.jl")

# The structured matrix types come from GeometricOptimizers, which also carries every method GML
# needs on them; this package defines none of its own. They are re-exported here, so that `using
# GeometricMachineLearning` on its own still gives a caller the matrix types its layers are
# parametrized by.
export SymmetricMatrix, SkewSymMatrix
export StrictlyLowerTriangular, StrictlyUpperTriangular
export StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix
export StiefelProjection

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

# `_diff`, `_norm` and `_vcat` are the `NamedTuple`/`(q, p)` arms of subtraction, the norm and
# concatenation, and none of them is exported: they are helpers of `src/reduced_system/` and
# `src/layers/`, not surface. Nothing named `_add` stands beside them, and nothing here adds a
# method to `AbstractNeuralNetworks.add!`: that would be piracy, and `GeometricOptimizers` already
# defines the three-argument `add!` on the structured matrix types against its own generic, for a
# caller that wants them.

export GradientLayerQ, GradientLayerP, ActivationLayerQ, ActivationLayerP, LinearLayerQ,
       LinearLayerP

# The manifolds are GeometricOptimizers' too, along with the geometry that goes with them.
export StiefelManifold, GrassmannManifold, Manifold
export rgrad, metric, check

include("layers/sympnets.jl")
include("layers/forcing_layer.jl")
include("layers/bias_layer.jl")
include("layers/resnet.jl")
include("layers/wide_resnet.jl")
include("layers/parametric_resnet_layer.jl")
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

export NetworkLoss, TransformerLoss, FeedForwardLoss, AutoEncoderLoss, ReducedLoss

#INCLUDE ARCHITECTURES
include("architectures/neural_network_integrator.jl")
include("architectures/resnet.jl")
# The two parametric layers and the architecture that pairs them with the exported `ResNet` are
# public and not exported, so that the export list does not grow with the parametric family.
public WideResNetLayer, ParametricResNetLayer, ParametricResNet
include("architectures/transformer_integrator.jl")
include("architectures/standard_transformer_integrator.jl")
include("architectures/sympnet.jl")
include("architectures/forced_sympnet.jl")
include("architectures/autoencoder.jl")
include("architectures/symplectic_autoencoder.jl")
include("architectures/psd.jl")
include("architectures/transformer_neural_network.jl")
include("architectures/volume_preserving_feedforward.jl")
include("architectures/volume_preserving_transformer.jl")

export SympNet, LASympNet, GSympNet
# The forced SympNet and the forcing layers are public and not exported; they are reached as
# `GeometricMachineLearning.ForcedSympNet` and the like.
public ForcedSympNet, ForcingLayer, ForcingLayerQ, ForcingLayerP, ForcingLayerQP
export ClassificationTransformer, ClassificationLayer
export VolumePreservingFeedForward
export SymplecticAutoencoder, PSDArch

# `solve!` is not here: it is `SimpleSolvers`', this package adds methods to it, and a caller who
# wants it takes it from there or from `GeometricOptimizers`.
export encoder, decoder

# The verb a caller of this package writes for a trajectory. `iterate` is `Base`'s function, and a
# network method on it takes over a name every `for` loop in the caller's own code resolves.
export integrate

include("parametric.jl")
include("loss/losses.jl")

export AbstractPullback, ZygotePullback
include("pullbacks/zygote_pullback.jl")

export DataLoader
export Batch, train!
include("data_loader/tensor_assign.jl")
include("data_loader/batch.jl")
include("data_loader/parametric_data_loader.jl")
include("data_loader/train.jl")
# after the loader, because a constructor of `ParametricResNet` takes a `ParametricDataLoader`
include("architectures/parametric_resnet.jl")

# The training of a network that takes the parameters of the system beside its input.
public apply_parametric, ParametricLoss, ParametricDataLoader

include("reduced_system/reduced_system.jl")

export HRedSys, reduction_error, projection_error, integrate_reduced_system,
       integrate_full_system

include("layers/linear_symplectic_attention.jl")
include("layers/symplectic_attention.jl")
include("layers/inverse_layers.jl")
include("architectures/linear_symplectic_transformer.jl")
include("architectures/symplectic_transformer.jl")
include("architectures/weight_tied_symplectic_autoencoder.jl")

export LinearSymplecticAttention, LinearSymplecticAttentionQ, LinearSymplecticAttentionP
export LinearSymplecticTransformer
export SymplecticAttention, SymplecticAttentionQ, SymplecticAttentionP
export SymplecticTransformer
export inverse_layer
export InverseGradientLayer, InverseGradientLayerQ, InverseGradientLayerP
export InverseSymplecticAttention, InverseSymplecticAttentionQ, InverseSymplecticAttentionP
export TransposePSDLayer
export WeightTiedSymplecticAutoencoder

include("map_to_cpu.jl")

export save, load
end
