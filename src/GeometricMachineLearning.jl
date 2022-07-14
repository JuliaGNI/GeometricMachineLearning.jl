module GeometricMachineLearning

    using LinearAlgebra

    include("activations/abstract_activation_function.jl")
    include("activations/identity.jl")

    export AbstractLayer
    export FeedForwardLayer, LinearFeedForwardLayer
    export ResidualLayer

    include("layers/abstract_layer.jl")
    include("layers/feed_forward_layer.jl")
    include("layers/residual_layer.jl")

    export AbstractNeuralNetwork
    export VanillaNeuralNetwork
    export Inverse

    include("networks/abstract_neural_network.jl")
    include("networks/inverse_neural_network.jl")
    include("networks/vanilla_neural_network.jl")

    export train!, apply!, jacobian!

end
