module GeometricMachineLearning

    using LinearAlgebra

    export AbstractLayer
    export FeedForwardLayer
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
