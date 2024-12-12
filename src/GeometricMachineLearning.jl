module GeometricMachineLearning

    using AbstractNeuralNetworks
    using BandedMatrices
    using ChainRulesCore
    using Distances
    using GeometricBase
    using GeometricIntegrators
    using GeometricSolutions: GeometricSolution, EnsembleSolution, DataSeries, StateVariable
    using GeometricEquations: ODEProblem, HODEProblem, ODEEnsemble, HODEEnsemble
    using KernelAbstractions
    using LinearAlgebra
    using NNlib
    using ProgressMeter
    using Random
    using Zygote
    using ForwardDiff
    using InteractiveUtils
    using TimerOutputs
    using LazyArrays
    import SymbolicNeuralNetworks: input_dimension, output_dimension, SymbolicPullback
    using SymbolicNeuralNetworks: derivative, _get_contents, _get_params
    using Symbolics: @variables, substitute
    import Symbolics.SymbolicUtils.Code: create_array

    import AbstractNeuralNetworks: Architecture, Model, AbstractExplicitLayer, AbstractExplicitCell, AbstractNeuralNetwork , NeuralNetwork, UnknownArchitecture
    import AbstractNeuralNetworks: Chain, GridCell
    import AbstractNeuralNetworks: Dense, Linear, Recurrent
    import AbstractNeuralNetworks: IdentityActivation, ZeroVector
    import AbstractNeuralNetworks: add!, update!
    import AbstractNeuralNetworks: layer
    import AbstractNeuralNetworks: initialparameters
    import AbstractNeuralNetworks: parameterlength
    import AbstractNeuralNetworks: GlorotUniform
    import AbstractNeuralNetworks: params, architecture, model, dim
    import AbstractNeuralNetworks: AbstractPullback, NetworkLoss, _compute_loss
    # export params, architetcure, model
    import GeometricIntegrators.Integrators: method, GeometricIntegrator
    import NNlib: σ, sigmoid, softmax
    import Base: iterate, eltype
    #import LogExpFunctions: softmax

    export CPU, GPU
    export Chain, NeuralNetwork
    export Dense, Linear
    export initialparameters
    export parameterlength
    export NeuralNetworkParameters
    
    export σ, sigmoid, softmax

    # from GeometricBase to print docs
    export description

    include("utils.jl")

    include("data_loader/data_loader.jl")

    # INCLUDE ARRAYS
    include("arrays/skew_symmetric.jl")
    include("arrays/symmetric.jl")
    include("arrays/poisson_tensor.jl")
    include("arrays/abstract_lie_algebra_horizontal.jl")
    include("arrays/stiefel_lie_algebra_horizontal.jl")
    include("arrays/grassmann_lie_algebra_horizontal.jl")
    include("arrays/triangular.jl")
    include("arrays/lower_triangular.jl")
    include("arrays/upper_triangular.jl")

    export SymmetricMatrix, PoissonTensor, SkewSymMatrix
    export StiefelLieAlgHorMatrix
    export SymplecticLieAlgMatrix, SymplecticLieAlgHorMatrix
    export GrassmannLieAlgHorMatrix
    export StiefelProjection, SymplecticProjection
    export LowerTriangular, UpperTriangular

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

    # this defines empty retraction type structs (doesn't rely on anything)
    include("optimizers/manifold_related/retraction_types.jl")
    

    # are these needed?
    export UnknownProblem, NothingFunction

    # + operation has been overloaded to work with NamedTuples!
    export _add, apply_toNT, split_and_flatten, add!
    
    # GPU specific operations
    export convert_to_dev, Device, CPUDevice

    export GradientLayerQ, GradientLayerP, ActivationLayerQ, ActivationLayerP, LinearLayerQ, LinearLayerP
    export Linear
    export ResidualLayer
    export LinearSymplecticLayerP, LinearSymplecticLayerQ
    export SymplecticStiefelLayer

    include("manifolds/abstract_manifold.jl")
    include("manifolds/stiefel_manifold.jl")
    # include("manifolds/symplectic_stiefel_manifold.jl")
    include("manifolds/grassmann_manifold.jl")

    include("arrays/stiefel_projection.jl")

    export StiefelManifold, SymplecticStiefelManifold, GrassmannManifold, Manifold
    export rgrad, metric

    include("optimizers/optimizer_method.jl")
    include("optimizers/optimizer_caches.jl")
    include("optimizers/optimizer.jl")
    include("optimizers/gradient_optimizer.jl")
    include("optimizers/momentum_optimizer.jl")        
    include("optimizers/adam_optimizer.jl")
    include("optimizers/adam_optimizer_with_learning_rate_decay.jl")
    include("optimizers/bfgs_cache.jl")
    include("optimizers/bfgs_optimizer.jl")
    include("optimizers/init_optimizer_cache.jl")

    include("optimizers/manifold_related/global_sections.jl")
    include("optimizers/manifold_related/modified_exponential.jl")
    include("optimizers/manifold_related/retractions.jl")

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

    # INCLUDE OPTIMIZERS
    export OptimizerMethod, AbstractCache
    export GradientOptimizer, GradientCache
    export MomentumOptimizer, MomentumCache
    export AdamOptimizerWithDecay
    export AdamOptimizer, AdamCache
    export AdamOptimizerWithDecay
    export BFGSOptimizer, BFGSCache

    export Optimizer
    export optimization_step!

    export GlobalSection, apply_section, apply_section!
    export global_rep
    export Geodesic, Cayley
    export geodesic, cayley
    export retraction
    # export ⊙², √ᵉˡᵉ, /ᵉˡᵉ, scalar_add
    export update!
    export check

    #INCLUDE ABSTRACT TRAINING integrator
    export AbstractTrainingMethod

    export loss_single #, loss
    
    export HnnTrainingMethod
    export LnnTrainingMethod
    export SympNetTrainingMethod
    
    include("training_method/abstract_training_method.jl")

    # INCLUDE DATA TRAINING STRUCTURE
    export AbstractDataShape, TrajectoryData, SampledData
    export get_length_trajectory, get_Δt, get_nb_point, get_nb_trajectory, get_data

    include("data/data_shape.jl")

    export AbstractDataSymbol
    export PositionSymbol, PhaseSpaceSymbol, DerivativePhaseSpaceSymbol, PosVeloAccSymbol, PosVeloSymbol
    export DataSymbol
    export can_reduce, symbols, symboldiff

    include("data/data_symbol.jl")

    # INCLUDE TRAINING INTEGRATOR

    export TrainingMethod
    export symbol, shape
    export min_length_batch
    
    
    include("training_method/training_method.jl")

     # INCLUDE DATA TRAINING STRUCTURE
    export AbstractTrainingData
    export TrainingData
    export shape, symbols, dim, noisemaker, data_symbols # , problem
    export reduce_symbols, reshape_intoSampledData
    export aresame
    
    include("data/data_training.jl")

    export get_batch, complete_batch_size, check_batch_size
    
    include("data/batch.jl")

    # INCLUDE BACKENDS
    export LuxBackend
    export NeuralNetwork
    export arch

    include("backends/backends.jl")
    include("backends/lux.jl")

    export NetworkLoss, TransformerLoss, FeedForwardLoss, AutoEncoderLoss, ReducedLoss, HNNLoss

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
    include("architectures/recurrent_neural_network.jl")
    include("architectures/LSTM_neural_network.jl")
    include("architectures/transformer_neural_network.jl")
    include("architectures/volume_preserving_feedforward.jl")
    include("architectures/volume_preserving_transformer.jl")

    export HamiltonianArchitecture
    export LagrangianNeuralNetwork
    export SympNet, LASympNet, GSympNet
    export RecurrentNeuralNetwork
    export LSTMNeuralNetwork
    export ClassificationTransformer, ClassificationLayer
    export VolumePreservingFeedForward
    export SymplecticAutoencoder, PSDArch
    export HamiltonianArchitecture, StandardHamiltonianArchitecture, GeneralizedHamiltonianArchitecture

    export solve!, encoder, decoder

    export train!, apply!, jacobian!
    export iterate

    export default_arch

    include("architectures/default_architecture.jl")

    include("loss/losses.jl")
    include("loss/hnn_loss.jl")

    export AbstractPullback, ZygotePullback, SymbolicPullback
    include("pullbacks/zygote_pullback.jl")
    include("pullbacks/symbolic_hnn_pullback.jl")

    export DataLoader, onehotbatch
    export Batch, optimize_for_one_epoch!
    include("data_loader/tensor_assign.jl")
    include("data_loader/matrix_assign.jl")
    include("data_loader/mnist_utils.jl")
    include("data_loader/batch.jl")
    include("data_loader/optimize.jl")

    export default_optimizer

    include("optimizers/default_optimizer.jl")

    # INCLUDE TRAINING parameters

    export TrainingParameters

    include("training/training_parameters.jl")

    # INCLUDE NEURALNET SOLUTION

    export SingleHistory
    export parameters, datashape
    export History
    export last, sizemax, nbtraining, show

    include("nnsolution/history.jl")

    export NeuralNetSolution
    export problem, tstep, history, size_history
    export set_sizemax_history
    
    include("nnsolution/neural_net_solution.jl")

    export EnsembleNeuralNetSolution
    export push!, merge!

    include("nnsolution/neural_net_solution_ensemble.jl")

    # INCLUDE TRAINING integrator

    export TrainingSet
    export parameters # , data

    include("training/training_set.jl")

    export EnsembleTraining
    export isnnShared, isParametersShared, isDataShared
    export parameters, data
    export push!, merge!, size

    include("training/ensemble_training.jl")

    include("training/nn_parameters_transformation.jl")

    export loss_gradient
    export train!

    include("training/train.jl")

    export SymplecticEuler
    export SymplecticEulerA, SymplecticEulerB
    export SEuler, SEulerA, SEulerB

    include("training_method/symplectic_euler.jl")

    export HnnExactMethod
    export ExactHnn

    include("training_method/hnn_exact_method.jl")

    export VariationalMethod
    export VariationalMidPointMethod
    export VariaMidPoint

    include("training_method/variational_method.jl")

    export LnnExactMethod
    export ExactLnn

    include("training_method/lnn_exact_method.jl")

    export BasicSympNetMethod
    export BasicSympNet

    include("training_method/sympnet_basic_method.jl")

    export default_method
    
    include("training/default_method.jl")


    # INCLUDE ASSERTION Function
    export matching
    include("training/matching.jl")


    # INCLUDE PROBLEMS
    export HNNProblem, LNNProblem

    include("integrator/problem_hnn.jl")
    include("integrator/problem_lnn.jl")
    
    # INCLUDE INTEGRATOR 
    export NeuralNetMethod
    export method

    include("integrator/abstract_neural_net_method.jl")

    # INCLUDE INTEGRATION METHOD
    export  SympNetMethod
    export integrate, integrate_step!

    include("integrator/sympnet_integrator.jl")
 
    include("reduced_system/reduced_system.jl")

    export HRedSys, reduction_error, projection_error, integrate_reduced_system, integrate_full_system

    include("layers/linear_symplectic_attention.jl")
    include("architectures/linear_symplectic_transformer.jl")

    export LinearSymplecticAttention, LinearSymplecticAttentionQ, LinearSymplecticAttentionP
    export LinearSymplecticTransformer

    include("map_to_cpu.jl")
end