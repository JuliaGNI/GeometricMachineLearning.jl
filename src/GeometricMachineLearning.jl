module GeometricMachineLearning

    using BandedMatrices
    using LinearAlgebra
    using NNlib
    using Random


    include("gradient.jl")
    include("training.jl")

    include("activations/abstract_activation_function.jl")
    include("activations/identity_activation.jl")

    export SymmetricMatrix, SymplecticMatrix

    include("arrays/add.jl")
    include("arrays/block_identity_lower.jl")
    include("arrays/block_identity_upper.jl")
    include("arrays/symmetric.jl")
    include("arrays/symplectic.jl")
    include("arrays/zero_vector.jl")

    export AbstractLayer
    export FeedForwardLayer, LinearFeedForwardLayer
    export Gradient
    export Linear
    export ResidualLayer
    export LinearSymplecticLayerP, LinearSymplecticLayerQ
    export SymplecticStiefelLayer

    include("layers/abstract_layer.jl")
    include("layers/feed_forward_layer.jl")
    include("layers/gradient.jl")
    include("layers/linear.jl")
    include("layers/residual_layer.jl")
    include("layers/linear_symplectic_layer.jl")
    include("layers/manifold_layer.jl")
    include("layers/symplectic_stiefel_layer.jl")


    include("architectures/architectures.jl")
    include("backends/backends.jl")

    export LuxBackend

    include("backends/lux.jl")

    # set default backend in NeuralNetwork constructor
    NeuralNetwork(arch::AbstractArchitecture; kwargs...) = NeuralNetwork(arch, LuxBackend(); kwargs...)

    export NeuralNetwork
    export HamiltonianNeuralNetwork

    include("architectures/hamiltonian_neural_network.jl")

    export AbstractNeuralNetwork
    export Inverse
    export VanillaNeuralNetwork

    include("networks/inverse_neural_network.jl")
    include("networks/vanilla_neural_network.jl")
    
    export train!, apply!, jacobian!

    include("optimizers/Abstract_optimizer.jl")
    include("optimizers/standard_optimizer.jl")
    include("optimizers/adam_optimizer.jl")
    include("optimizers/momentum_optimizer.jl")

    export StandardOptimizer
    export AdamOptimizer
    export MomentumOptimizer
    export init
    export init_adam
    export init_momentum
    export setup
end
