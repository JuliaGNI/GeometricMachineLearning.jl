module GeometricMachineLearning

    using BandedMatrices
    using Distances
    using LinearAlgebra
    using NNlib
    using ProgressMeter
    using Random
    using Zygote

    import Lux

    export TrivialInitRNG


    include("rng/trivial_rng.jl")


    include("gradient.jl")
    include("training.jl")

    #+ operation has been overloaded to work with NamedTuples!
    export _add

    include("utils.jl")

    include("activations/abstract_activation_function.jl")
    include("activations/identity_activation.jl")

    export SymmetricMatrix, SymplecticMatrix, SkewSymMatrix
    export StiefelLieAlgHorMatrix
    export SymplecticLieAlgMatrix, SymplecticLieAlgHorMatrix
    export StiefelProjection, SymplecticProjection

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

    export AbstractLayer
    export FeedForwardLayer, LinearFeedForwardLayer
    export Gradient
    export Linear
    export ResidualLayer
    export LinearSymplecticLayerP, LinearSymplecticLayerQ
    export SymplecticStiefelLayer
    export StiefelLayer, ManifoldLayer
    export AbstractNeuralNetwork

    export retraction
    export GlobalSection
    export global_rep

    include("manifolds/stiefel_manifold.jl")
    include("manifolds/symplectic_manifold.jl")
    include("manifolds/abstract_manifold.jl")

    include("layers/abstract_layer.jl")
    include("layers/feed_forward_layer.jl")
    include("layers/gradient.jl")
    include("layers/linear.jl")
    include("layers/residual_layer.jl")
    include("layers/linear_symplectic_layer.jl")
    include("layers/manifold_layer.jl")
    include("optimizers/retraction_types.jl")
    include("layers/stiefel_layer.jl")
    include("optimizers/retractions.jl")


    include("optimizers/global_sections.jl")
    include("optimizers/optimizer_layer_caches.jl")
    include("optimizers/abstract_optimizer.jl")
    #include("optimizers/standard_optimizer.jl")
    #include("optimizers/momentum_optimizer.jl")
    #include("optimizers/adam_optimizer.jl")
    #include("optimizers/optimizer_cache.jl")

    export AbstractOptimizer, AbstractLayerCache
    export GradientOptimizer, StandardLayerCache
    export MomentumOptimizer, MomentumLayerCache
    export AdamOptimizer, AdamLayerCache

    export Optimiser, AbstractMethodOptimiser
    include("optimizers/method_optimizer.jl")
    include("optimizers/optimiser.jl")
   
    


    export AbstractNeuralNetwork

    include("architectures/architectures.jl")
    include("backends/backends.jl")

    export LuxBackend

    include("backends/lux.jl")

    # set default backend in NeuralNetwork constructor
    NeuralNetwork(arch::AbstractArchitecture; kwargs...) = NeuralNetwork(arch, LuxBackend(); kwargs...)

    export NeuralNetwork
    export HamiltonianNeuralNetwork
    export LagrangianNeuralNetwork
    export SympNet
    export LASympNet
    export GSympNet

    include("architectures/autoencoder.jl")
    include("architectures/fixed_width_network.jl")
    include("architectures/hamiltonian_neural_network.jl")
    include("architectures/lagrangian_neural_network.jl")
    include("architectures/variable_width_network.jl")
    include("architectures/sympnet.jl")

    export train!, apply!, jacobian!
    export Iterate_Sympnet

    export ⊙², /ᵉˡᵉ, scalar_add, √ᵉˡᵉ

    export update!
    export check
    export init_optimizer_cache
    export optimization_step!

end
