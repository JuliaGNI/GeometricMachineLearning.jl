module GeometricMachineLearning

    using AbstractNeuralNetworks
    using BandedMatrices
    using ChainRulesCore
    using Distances
    using ForwardDiff
    using GeometricBase
    using GeometricEquations
    using GeometricIntegrators
    using KernelAbstractions
    using LinearAlgebra
    using NNlib
    using ProgressMeter
    using Random
    using Zygote
    using ForwardDiff
    using InteractiveUtils
    using TimerOutputs

    import AbstractNeuralNetworks: Architecture, Model, AbstractExplicitLayer, AbstractExplicitCell, AbstractNeuralNetwork , NeuralNetwork
    import AbstractNeuralNetworks: Chain, GridCell
    import AbstractNeuralNetworks: Dense, Linear, Recurrent
    import AbstractNeuralNetworks: IdentityActivation, ZeroVector
    import AbstractNeuralNetworks: add!, update!
    import AbstractNeuralNetworks: layer
    import AbstractNeuralNetworks: initialparameters
    import AbstractNeuralNetworks: parameterlength
    import AbstractNeuralNetworks: GlorotUniform
    import AbstractNeuralNetworks: params, architecture, model, dim
    # export params, architetcure, model
    export dim
    import GeometricIntegrators.Integrators: method, GeometricIntegrator
    import NNlib: σ, sigmoid, softmax
    #import LogExpFunctions: softmax

    export CPU, GPU
    export Chain, NeuralNetwork
    export Dense, Linear
    export initialparameters
    export parameterlength
    
    export σ, sigmoid, softmax

    # from GeometricBase to print docs
    export description

    # the functionality in the script doesn't require anything else defined in GML, but some of the other scripts in that folder do.
    include("data_loader/data_loader.jl")

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
    include("kernels/inverses/inverse_kernel.jl")
    include("kernels/vec_tensor_mul.jl")

    include("kernels/kernel_ad_routines/assign_q_and_p.jl")
    include("kernels/kernel_ad_routines/tensor_mat_mul.jl")
    include("kernels/kernel_ad_routines/mat_tensor_mul.jl")
    include("kernels/kernel_ad_routines/tensor_tensor_mul.jl")
    include("kernels/kernel_ad_routines/tensor_transpose_tensor_mul.jl")
    include("kernels/kernel_ad_routines/tensor_transpose.jl")
    include("kernels/kernel_ad_routines/vec_tensor_mul.jl")
    # export tensor_mat_mul

    # this defines empty retraction type structs (doesn't rely on anything)
    include("optimizers/manifold_related/retraction_types.jl")
    

    # are these needed?
    export UnknownProblem, NothingFunction
    include("utils.jl")

    # + operation has been overloaded to work with NamedTuples!
    export _add, apply_toNT, split_and_flatten, add!
    
    # GPU specific operations
    export convert_to_dev, Device, CPUDevice

    # INCLUDE ARRAYS
    include("arrays/skew_symmetric.jl")
    include("arrays/symmetric.jl")
    include("arrays/symplectic.jl")
    include("arrays/abstract_lie_algebra_horizontal.jl")
    include("arrays/stiefel_lie_algebra_horizontal.jl")
    include("arrays/grassmann_lie_algebra_horizontal.jl")

    export SymmetricMatrix, SymplecticPotential, SkewSymMatrix
    export StiefelLieAlgHorMatrix
    export SymplecticLieAlgMatrix, SymplecticLieAlgHorMatrix
    export GrassmannLieAlgHorMatrix
    export StiefelProjection, SymplecticProjection

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
    include("layers/attention_layer.jl")
    include("layers/transformer.jl")
    include("layers/psd_like_layer.jl")
    include("layers/classification.jl")

    # include("layers/symplectic_stiefel_layer.jl")
    export StiefelLayer, GrassmannLayer, ManifoldLayer
    export PSDLayer
    export MultiHeadAttention
    export Attention
    export ResNet
    export Transformer
    export Classification

    # INCLUDE OPTIMIZERS
    export OptimizerMethod, AbstractCache
    export GradientOptimizer, GradientCache
    export MomentumOptimizer, MomentumCache
    export AdamOptimizer, AdamCache
    export BFGSOptimizer, BFGSCache

    export Optimizer
    export optimization_step!

    export GlobalSection, apply_section
    export global_rep
    export Geodesic, Cayley
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

    export DataLoader, onehotbatch, accuracy
    export Batch, optimize_for_one_epoch!
    include("data_loader/tensor_assign.jl")
    include("data_loader/matrix_assign.jl")
    include("data_loader/mnist_utils.jl")
    include("data_loader/batch.jl")

    #INCLUDE ARCHITECTURES
    include("architectures/sympnet.jl")
    include("architectures/autoencoder.jl")
    include("architectures/fixed_width_network.jl")
    include("architectures/hamiltonian_neural_network.jl")
    include("architectures/lagrangian_neural_network.jl")
    include("architectures/variable_width_network.jl")
    include("architectures/recurrent_neural_network.jl")
    include("architectures/LSTM_neural_network.jl")
    include("architectures/transformer_neural_network.jl")

    export HamiltonianNeuralNetwork
    export LagrangianNeuralNetwork
    export SympNet
    export LASympNet
    export GSympNet
    export RecurrentNeuralNetwork
    export LSTMNeuralNetwork
    export ClassificationTransformer

    export train!, apply!, jacobian!
    export Iterate_Sympnet

    export default_arch

    include("architectures/default_architecture.jl")

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
    export nn, problem, tstep, history, size_history
    export set_sizemax_history
    
    include("nnsolution/neural_net_solution.jl")

    export EnsembleNeuralNetSolution
    export push!, merge!

    include("nnsolution/neural_net_solution_ensemble.jl")

    # INCLUDE TRAINING integrator

    export TrainingSet
    export nn, parameters # , data

    include("training/training_set.jl")

    export EnsembleTraining
    export isnnShared, isParametersShared, isDataShared
    export nn, parameters, data
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
 
    include("reduced_system/system_type.jl")
    include("reduced_system/reduced_system.jl")

    export ReducedSystem, compute_reduction_error, compute_projection_error, reduced_vector_field_from_full_explicit_vector_field, perform_integration_reduced, perform_integration_full
end