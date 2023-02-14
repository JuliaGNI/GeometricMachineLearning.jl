module GeometricMachineLearning

    using BandedMatrices
    using LinearAlgebra


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
    export ResidualLayer
    export LinearSymplecticLayerP, LinearSymplecticLayerQ
    export SymplecticStiefelLayer

    include("layers/abstract_layer.jl")
    include("layers/feed_forward_layer.jl")
    include("layers/gradient.jl")
    include("layers/residual_layer.jl")
    include("layers/linear_symplectic_layer.jl")
    include("layers/manifold_layer.jl")

    export AbstractNeuralNetwork
    export HamiltonianNeuralNetwork
    export Inverse
    export VanillaNeuralNetwork

    include("networks/abstract_neural_network.jl")
    include("networks/inverse_neural_network.jl")
    include("networks/vanilla_neural_network.jl")
    
    export train!, apply!, jacobian!

    include("optimizers/AbstractOptimizer.jl")
    include("optimizers/StandardOptimizer.jl")
    include("optimizers/AdamOptimizer.jl")
    include("optimizers/MomentumOptimizer.jl")

    export StandardOptimizer
    export AdamOptimizer
    export MomentumOptimizer
    export init
    export init_adam
    export init_momentum
    export setup
end
