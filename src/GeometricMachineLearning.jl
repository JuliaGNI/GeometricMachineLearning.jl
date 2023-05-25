module GeometricMachineLearning

    using BandedMatrices
    using Distances
    using LinearAlgebra
    using NNlib
    using ProgressMeter
    using Random
    using Zygote

    import Lux

    #this defines empty retraction type structs (doesn't rely on anything)
    include("optimizers/retraction_types.jl")


    include("rng/trivial_rng.jl")

    export TrivialInitRNG

    #are these needed?
    include("gradient.jl")
    include("training.jl")
    include("utils.jl")

    #+ operation has been overloaded to work with NamedTuples!
    export _add, apply_toNT, split_and_flatten, add!

    include("embeddings/sin_cos.jl")

    export sc_embed

    include("activations/abstract_activation_function.jl")
    include("activations/identity_activation.jl")


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

    export SymmetricMatrix, SymplecticMatrix, SkewSymMatrix
    export StiefelLieAlgHorMatrix
    export SymplecticLieAlgMatrix, SymplecticLieAlgHorMatrix
    export StiefelProjection, SymplecticProjection

    export AbstractLayer
    export FeedForwardLayer, LinearFeedForwardLayer
    export Gradient
    export Linear
    export ResidualLayer
    export LinearSymplecticLayerP, LinearSymplecticLayerQ
    export SymplecticStiefelLayer

    include("manifolds/abstract_manifold.jl")
    include("manifolds/stiefel_manifold.jl")
    include("manifolds/symplectic_manifold.jl")

    export StiefelManifold, SymplecticStiefelManifold, Manifold

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
    include("layers/embedding.jl")


    #include("layers/symplectic_stiefel_layer.jl")
    export StiefelLayer, ManifoldLayer
    export MultiHeadAttention
    export Transformer
    export AbstractNeuralNetwork
    export retraction
    export Embedding

    include("architectures/architectures.jl")
    include("backends/backends.jl")
    include("backends/normal_forms_workaround.jl")

    export LuxBackend

    include("backends/lux.jl")

    # set default backend in NeuralNetwork constructor
    NeuralNetwork(arch::AbstractArchitecture; kwargs...) = NeuralNetwork(arch, LuxBackend(); kwargs...)

    export NeuralNetwork
    export HamiltonianNeuralNetwork

    include("architectures/autoencoder.jl")
    include("architectures/fixed_width_network.jl")
    include("architectures/hamiltonian_neural_network.jl")
    include("architectures/variable_width_network.jl")

    export train!, apply!, jacobian!

    include("optimizers/global_sections.jl")
    include("optimizers/optimizer_caches.jl")
    include("optimizers/abstract_optimizer.jl")
    include("optimizers/standard_optimizer.jl")
    include("optimizers/momentum_optimizer.jl")
    include("optimizers/adam_optimizer.jl")
    include("optimizers/auxiliary.jl")
    include("optimizers/retractions.jl")

    export GlobalSection, apply_section
    export global_rep

    export TrivialInitRNG
    
    export AbstractOptimizer, AbstractCache
    export StandardOptimizer, StandardCache
    export MomentumOptimizer, MomentumCache
    export AdamOptimizer, AdamCache

    export Geodesic, Cayley
    export retraction

    export ⊙², √ᵉˡᵉ, /ᵉˡᵉ, scalar_add

    export update!
    export check
    export init_optimizer_cache
    export optimization_step!

    include("rng/random_funcs.jl")

end
