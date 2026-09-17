module GeometricMachineLearning

using AbstractNeuralNetworks
# The parameter container lives in `NeuralNetworkParameters` as of `AbstractNeuralNetworks` 0.7,
# under the name `NetworkParameters`. The import is selective rather than a bare `using`: that
# package also exports `flatten`/`unflatten` and the leaf protocol, none of which this package
# extends — `GeometricOptimizers` carries the protocol for the structured matrices.
#
# Two things do come from there rather than being written again here: `mapstorage`, which reaches the
# storage of a structured leaf and rebuilds the leaf around the result (`src/map_to_cpu.jl`), and
# `parameter_eltype`, which promotes over the leaves of a set. A plain `NamedTuple` of layers is
# walked with `Base.map`, which needs nothing from anybody.
import NeuralNetworkParameters: NetworkParameters
using NeuralNetworkParameters: mapparameters, mapstorage, parameter_eltype
using ChainRulesCore
using GeometricBase
using GeometricSolutions: GeometricSolution, EnsembleSolution, DataSeries, StateVariable,
                          TimeSeries
using GeometricEquations: EnsembleProblem, ODEProblem, HODEProblem, ODEEnsemble,
                          HODEEnsemble
using KernelAbstractions
using LinearAlgebra
using NNlib
using ProgressMeter
using Random
using Zygote
using ForwardDiff
using InteractiveUtils
using TimerOutputs
import SymbolicNeuralNetworks
import SymbolicNeuralNetworks: SymbolicPullback
using SymbolicNeuralNetworks: derivative, SymbolicNeuralNetwork
import Symbolics

# The manifolds, the structured matrix types, the global sections and the retractions are
# `GeometricOptimizers`' — GML used to carry near-verbatim copies of all eleven types, which Julia
# saw as *distinct* from the upstream ones, so none of GeometricOptimizers' generic machinery
# dispatched on them and GML re-implemented the retraction pipeline four times over. See
# [#234](https://github.com/JuliaGNI/GeometricMachineLearning.jl/issues/234).
#
# `import` and not `using ...: ...`: GML adds constructor methods to several of these types (in
# `layers/`, `arrays/gml_extensions.jl` and the kernels), and extending a *type* reached through
# `using` warns on every such method since Julia 1.12 — "Constructor for type … was extended in
# `GeometricMachineLearning` without explicit qualification or import". Everything imported here is
# re-exported below, which is what keeps `using GeometricMachineLearning` alone sufficient.
import GeometricOptimizers
import GeometricOptimizers: Manifold, StiefelManifold, GrassmannManifold
import GeometricOptimizers: SkewSymMatrix, SymmetricMatrix, AbstractTriangular,
                            LowerTriangular, UpperTriangular, StiefelProjection
import GeometricOptimizers: AbstractLieAlgHorMatrix, StiefelLieAlgHorMatrix,
                            GrassmannLieAlgHorMatrix
import GeometricOptimizers: rgrad, metric, check, Ω, global_section
# `assign_columns(Q, N, n)` — the first `n` columns of a `QR` factor, allocated on `Q`'s backend.
# It is upstream's, and internal there; the three manifold layers below initialise their weights
# with it.
import GeometricOptimizers: assign_columns
import GeometricOptimizers: GlobalSection, global_rep, apply_section, apply_section!,
                            update_section!
import GeometricOptimizers: AbstractRetraction, Geodesic, Cayley, geodesic, cayley,
                            retraction
import GeometricOptimizers: OptimizerMethod, OptimizerSolution,
                            GradientMethod, MomentumMethod, Adam,
                            GradientState, MomentumState, AdamState,
                            AdamOptimizerWithDecay, DecayingStatic
import GeometricOptimizers: update!
# `solve!` is imported rather than started afresh so that GML's `solve!(::NeuralNetwork{<:PSDArch},
# …)` — solve for the parameters directly, by SVD, instead of training for them — is a method of the
# same verb a caller already has from GeometricOptimizers, and not a second function of the name.
import GeometricOptimizers: solve!
# The optimizer *caches* stay internal upstream — they are `solver_step!` scratch — so GML reaches
# them as `GeometricOptimizers.AdamCache` where it needs to name one, and no longer re-exports them.

import AbstractNeuralNetworks: Architecture, Model, AbstractExplicitLayer,
                               AbstractNeuralNetwork, NeuralNetwork,
                               UnknownArchitecture, FeedForwardLoss
import AbstractNeuralNetworks: Chain
# `input_dimension`/`output_dimension` are AbstractNeuralNetworks' since v0.6.4; the `Chain`
# methods GML uses are added to them by SymbolicNeuralNetworks.
import AbstractNeuralNetworks: input_dimension, output_dimension
import AbstractNeuralNetworks: Dense, Linear
# `update!` used to be imported here too, from `AbstractNeuralNetworks`, and re-exported. GML never
# added a method to it, so all the export did was shadow `GeometricOptimizers.update!` — which is
# `GeometricBase.update!`, a different generic function, and the one that actually has methods for
# the optimizer caches. It is imported from GeometricOptimizers with the rest of them below.
import AbstractNeuralNetworks: add!
import AbstractNeuralNetworks: layer
import AbstractNeuralNetworks: initialparameters
import AbstractNeuralNetworks: parameterlength
import AbstractNeuralNetworks: GlorotUniform
import AbstractNeuralNetworks: params, architecture, model, dim
import AbstractNeuralNetworks: AbstractPullback, NetworkLoss, _compute_loss
import AbstractNeuralNetworks: networkbackend
# `save` and `load` are `NeuralNetworkParameters`' generics; `AbstractNeuralNetworks` 0.7 only
# re-binds them. Reach for them where they are defined.
import NeuralNetworkParameters: save, load
# export params, architetcure, model
export dim
import NNlib: σ, sigmoid, softmax
import Base: iterate, eltype
#import LogExpFunctions: softmax

export CPU, GPU
export Chain, NeuralNetwork
export Dense, Linear
export initialparameters
export parameterlength
export NetworkParameters

export σ, sigmoid, softmax

# `GeometricBase` defines `description` but does not export it, so `using GeometricBase` alone does
# not bring it into scope and re-exporting it needs the explicit import.
import GeometricBase: description
export description

include("utils.jl")

include("data_loader/data_loader.jl")

# INCLUDE ARRAYS — the structured matrix types come from GeometricOptimizers; `PoissonTensor` is
# GML's own, and `gml_extensions.jl` holds what GML adds to the upstream types.
include("arrays/poisson_tensor.jl")
include("arrays/gml_extensions.jl")

# Re-exported from GeometricOptimizers, so that `using GeometricMachineLearning` on its own still
# gives a caller the matrix types its layers are parametrized by.
export SymmetricMatrix, SkewSymMatrix
export LowerTriangular, UpperTriangular
export StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix
export StiefelProjection
# GML's own
export PoissonTensor
# `SymplecticLieAlgMatrix`, `SymplecticLieAlgHorMatrix` and `SymplecticProjection` used to be
# exported here. Nothing has defined them for as long as the git history goes back, so the exports
# were silent `UndefVarError`s waiting for a caller. GML has no test that would have caught them —
# ten exported names are still undefined, which is issue C10; `GeometricOptimizers`' own
# `test/exports.jl` is the one-assertion-over-`names` shape that closes this class.

include("kernels/assign_q_and_p.jl")
include("kernels/tensor_mat_mul.jl")
include("kernels/tensor_tensor_mul.jl")
include("kernels/tensor_transpose_tensor_mul.jl")
include("kernels/tensor_tensor_transpose_mul.jl")
include("kernels/tensor_transpose_mat_mul.jl")
include("kernels/tensor_transpose_tensor_transpose_mul.jl")
include("kernels/mat_tensor_mul.jl")
include("kernels/tensor_transpose.jl")
include("kernels/exponentials/tensor_exponential.jl")
include("kernels/inverses/cpu_inverse.jl")
include("kernels/inverses/inverse_2x2.jl")
include("kernels/inverses/inverse_3x3.jl")
include("kernels/inverses/inverse_4x4.jl")
include("kernels/inverses/inverse_5x5.jl")
include("kernels/inverses/tensor_cayley.jl")
include("kernels/inverses/tensor_mat_skew_sym_assign.jl")
include("kernels/vec_tensor_mul.jl")

include("kernels/kernel_ad_routines/matrix_cotangent.jl")
include("kernels/kernel_ad_routines/assign_q_and_p.jl")
include("kernels/kernel_ad_routines/tensor_mat_mul.jl")
include("kernels/kernel_ad_routines/mat_tensor_mul.jl")
include("kernels/kernel_ad_routines/tensor_tensor_mul.jl")
include("kernels/kernel_ad_routines/tensor_transpose_mat_mul.jl")
include("kernels/kernel_ad_routines/tensor_transpose_tensor_mul.jl")
include("kernels/kernel_ad_routines/tensor_transpose.jl")
include("kernels/kernel_ad_routines/tensor_mat_skew_sym_assign.jl")
include("kernels/kernel_ad_routines/vec_tensor_mul.jl")
# export tensor_mat_mul

export MatrixSoftmax, VectorSoftmax
include("activations/softmax.jl")

# are these needed?
export UnknownProblem, NothingFunction

# `_add`, `_diff` and `_norm` are the `NamedTuple`/`(q, p)` arms of addition, subtraction and the
# norm, and none of the three is exported: they are helpers of `src/reduced_system/`, not surface.
# `_add` was the odd one out until 0.7.0, as was `add!` -- which is `AbstractNeuralNetworks`' generic
# and available from there, this package only adding methods for the structured matrix types.

export GradientLayerQ, GradientLayerP, ActivationLayerQ, ActivationLayerP, LinearLayerQ,
       LinearLayerP
export Linear
# `SymplecticStiefelLayer`, `ResidualLayer`, `LinearSymplecticLayerP`, `LinearSymplecticLayerQ`,
# `convert_to_dev`, `Device` and `CPUDevice` were exported here and defined nowhere. The layer of
# `ResidualLayer`'s shape that this package loads is `ResNetLayer`; the device operations have no
# implementation at all.

# The manifolds are GeometricOptimizers' too, along with the geometry that goes with them.
export StiefelManifold, GrassmannManifold, Manifold
export rgrad, metric, check

include("layers/sympnets.jl")
include("layers/bias_layer.jl")
include("layers/resnet.jl")
include("layers/manifold_layer.jl")
include("layers/stiefel_layer.jl")
include("layers/grassmann_layer.jl")
include("layers/multi_head_attention.jl")
include("layers/volume_preserving_attention.jl")
include("layers/volume_preserving_feedforward.jl")
include("layers/transformer.jl")
include("layers/psd_like_layer.jl")
include("layers/classification.jl")

# include("layers/symplectic_stiefel_layer.jl")
export StiefelLayer, GrassmannLayer, ManifoldLayer
export PSDLayer
export MultiHeadAttention
export VolumePreservingAttention
export VolumePreservingFeedForwardLayer
export VolumePreservingLowerLayer
export VolumePreservingUpperLayer
export VolumePreservingTransformer
export NeuralNetworkIntegrator
export ResNet
export Transformer
export TransformerIntegrator, StandardTransformerIntegrator

# INCLUDE OPTIMIZERS — the methods, states, sections and retractions come from GeometricOptimizers.
# `go_bridges.jl` used to sit here: thirty-odd methods reconnecting GML's copies of the structured
# types to GeometricOptimizers' `_add!`/`_rac!`/`_square!`/`_div!`/`_rmul!`/`update_section!`. The
# types are the same objects now, so upstream's own methods apply and the file is gone.
include("optimizers/optimizer.jl")

export OptimizerMethod
export GradientMethod, GradientState
export MomentumMethod, MomentumState
export Adam, AdamState
export Optimizer
export optimization_step!
export GlobalSection, global_section, apply_section, apply_section!, update_section!
export global_rep
export Geodesic, Cayley
export geodesic, cayley
export retraction
export update!
# `AbstractCache` and the three cache types used to be exported here. The caches are
# `solver_step!` scratch and stay internal to GeometricOptimizers, for every method alike; reach one
# as `GeometricOptimizers.AdamCache` if you genuinely need to name it.
# backward-compat aliases (old names → new names)
const GradientOptimizer = GradientMethod
const MomentumOptimizer = MomentumMethod
const AdamOptimizer = Adam
export GradientOptimizer, MomentumOptimizer, AdamOptimizer
# Re-exported from GeometricOptimizers, which owns the one definition of them now. GML's own
# `AdamOptimizerWithDecay` was a second, incompatible export of the same name — issue B1.
export AdamOptimizerWithDecay, DecayingStatic

export NeuralNetwork

export NetworkLoss, TransformerLoss, FeedForwardLoss, AutoEncoderLoss, ReducedLoss, HNNLoss,
       LNNLoss, SymplecticEulerLoss, VariationalMidpointLoss

#INCLUDE ARCHITECTURES
include("architectures/neural_network_integrator.jl")
include("architectures/resnet.jl")
include("architectures/transformer_integrator.jl")
include("architectures/standard_transformer_integrator.jl")
include("architectures/sympnet.jl")
include("architectures/autoencoder.jl")
include("architectures/symplectic_autoencoder.jl")
include("architectures/psd.jl")
include("architectures/fixed_width_network.jl")
include("architectures/hamiltonian_neural_network.jl")
include("architectures/lagrangian_neural_network.jl")
include("architectures/variable_width_network.jl")
include("architectures/transformer_neural_network.jl")
include("architectures/volume_preserving_feedforward.jl")
include("architectures/volume_preserving_transformer.jl")

export HamiltonianArchitecture
export LagrangianNeuralNetwork
export SympNet, LASympNet, GSympNet
export ClassificationTransformer, ClassificationLayer
export VolumePreservingFeedForward
export SymplecticAutoencoder, PSDArch
export HamiltonianArchitecture, StandardHamiltonianArchitecture,
       GeneralizedHamiltonianArchitecture

export solve!, encoder, decoder

export iterate

include("loss/losses.jl")
include("loss/hnn_loss.jl")
include("loss/lnn_loss.jl")
include("loss/symplectic_euler_loss.jl")
include("loss/variational_midpoint_loss.jl")

export AbstractPullback, ZygotePullback, SymbolicPullback
include("pullbacks/zygote_pullback.jl")
include("pullbacks/symbolic_hnn_pullback.jl")

export DataLoader
export Batch, optimize_for_one_epoch!
include("data_loader/tensor_assign.jl")
include("data_loader/matrix_assign.jl")
include("data_loader/batch.jl")
include("data_loader/optimize.jl")

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
