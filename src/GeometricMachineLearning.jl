module GeometricMachineLearning

    using BandedMatrices
    using Distances
    using LinearAlgebra
    using NNlib
    using ProgressMeter
    using Random
    using Zygote
    using ForwardDiff

    import Lux

    #this defines empty retraction type structs (doesn't rely on anything)
    include("optimizers/useful_functions/retraction_types.jl")


    export TrivialInitRNG

    include("rng/trivial_rng.jl")
    

    #are these needed?
    include("gradient.jl")
    include("training.jl")
    include("utils.jl")

    #+ operation has been overloaded to work with NamedTuples!
    export _add, apply_toNT, split_and_flatten, add!

    #+ operation has been overloaded to work with NamedTuples!
    export _add

    include("activations/abstract_activation_function.jl")
    include("activations/identity_activation.jl")


    #INCLUDE ARRAYS
    include("arrays/add.jl")
    include("arrays/zero_vector.jl")
    include("arrays/block_identity_lower.jl")
    include("arrays/block_identity_upper.jl")
    include("arrays/symmetric.jl")
    include("arrays/symplectic.jl")
    include("arrays/symplectic_lie_alg.jl")
    include("arrays/sympl_lie_alg_hor.jl")
    include("arrays/skew_sym.jl")
    include("arrays/stiefel_lie_alg_hor.jl")
    include("arrays/auxiliary.jl")

    export retraction
    export GlobalSection
    export global_rep


    #INCLUDE MANIFOLDS
    include("manifolds/abstract_manifold.jl")
    include("manifolds/stiefel_manifold.jl")
    include("manifolds/symplectic_manifold.jl")

    export Manifold
    export StiefelManifold
    export SymplecticStiefelManifold
    
    #INCLUDE LAYERS
    #include("layers/symplectic_stiefel_layer.jl")
    export AbstractLayer
    export FeedForwardLayer, LinearFeedForwardLayer
    export Gradient
    export Linear
    export ResidualLayer
    export LinearSymplecticLayerP, LinearSymplecticLayerQ
    export SymplecticStiefelLayer
    export StiefelLayer, ManifoldLayer
    
    export MultiHeadAttention
    export Transformer
    export AbstractNeuralNetwork
    export retraction

    include("layers/abstract_layer.jl")
    include("layers/feed_forward_layer.jl")
    include("layers/gradient.jl")
    include("layers/linear.jl")
    include("layers/resnet.jl")
    include("layers/linear_symplectic_layer.jl")
    include("layers/manifold_layer.jl")
    include("layers/stiefel_layer.jl")
    include("layers/multi_head_attention.jl")
    include("layers/transformer.jl")


    #INCLUDE OPTIMIZERS
    export AbstractMethodOptimiser, AbstractCache
    export GradientOptimizer, GradientCache
    export MomentumOptimizer, MomentumCache
    export AdamOptimizer, AdamCache

    export Optimiser
    export optimization_step!
    export init_optimizer_cache

    include("optimizers/optimizer_caches.jl")
    include("optimizers/Method_Optimizer/abstract_method_optimizer.jl")
    include("optimizers/Method_Optimizer/gradient_optimizer.jl")
    include("optimizers/Method_Optimizer/momentum_optimizer.jl")        
    include("optimizers/Method_Optimizer/adam_optimizer.jl")
    include("optimizers/optimizer.jl")

    export GlobalSection, apply_section
    export global_rep
    export TrivialInitRNG
    export Geodesic, Cayley
    export retraction
    #export ⊙², √ᵉˡᵉ, /ᵉˡᵉ, scalar_add
    export update!
    export check

    include("optimizers/useful_functions/global_sections.jl")
    include("optimizers/useful_functions/auxiliary.jl")
    include("optimizers/useful_functions/retractions.jl")


    #ASSERTION FUNCTION 
    export assert

    #INCLUDE DATA TRAINING STRUCTURE
    export AbstractTrainingData
    export DataTrajectory, DataSampled, DataTarget
    
    include("data/data_training.jl")
    include("data/batch.jl")


    #INCLUDE BACKENDS
    export AbstractNeuralNetwork
    export LuxBackend
    export NeuralNetwork

    include("architectures/architectures.jl")
    include("backends/backends.jl")
    include("backends/lux.jl")

    # set default backend in NeuralNetwork constructor
    NeuralNetwork(arch::AbstractArchitecture; kwargs...) = NeuralNetwork(arch, LuxBackend(); kwargs...)

    
    export Hnn_training_integrator
    export Lnn_training_integrator
    export SEuler
    export ExactIntegrator
    export ExactIntegratorLNN
    export VariationalMidPointLNN
    export SympNetIntegrator
    export BaseIntegrator

    #INCLUDE ARCHITECTURES
    export HamiltonianNeuralNetwork
    export LagrangianNeuralNetwork
    export SympNet
    export LASympNet
    export GSympNet

    export train!, apply!, jacobian!
    export Iterate_Sympnet

    include("architectures/architectures.jl")
    include("architectures/autoencoder.jl")
    include("architectures/fixed_width_network.jl")
    include("architectures/hamiltonian_neural_network.jl")
    include("architectures/lagrangian_neural_network.jl")
    include("architectures/variable_width_network.jl")
    include("architectures/sympnet.jl")

    #INCLUDE TRAINING integrator

    export AbstractTrainingIntegrator
    export HnnTrainingIntegrator
    export LnnTrainingIntegrator
    export SympNetTrainingIntegrator

    export TrainingIntegrator

    export loss_single, loss, loss_gradient
    export train!

    include("training/train.jl")

    export SymplecticEuler
    export _SymplecticEulerA, _SymplecticEulerB
    export SymplecticEulerA, SymplecticEulerB

    include("training/hnn_training/symplectic_euler.jl")

    export HnnExactIntegrator

    include("training/hnn_training/hnn_exact_integrator.jl")

    export VariationalIntegrator
    export VariationalMidPointIntegrator

    include("training/lnn_training/variational_integrator.jl")

    export LnnExactIntegrator

    include("training/lnn_training/lnn_exact_integrator.jl")

    export BasicSympNetIntegrator

    include("training/sympnet_training/sympnet_basic_integrator.jl")

    export default_integrator
    
    include("training/default_integrator.jl")





    include("rng/random_funcs.jl")

    
end
