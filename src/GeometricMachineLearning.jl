module GeometricMachineLearning

    using BandedMatrices
    using Distances
    using LinearAlgebra
    using NNlib
    using ProgressMeter
    using Random
    using Zygote
    using KernelAbstractions
    using CUDAKernels

    import Lux, CUDA

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
    
    #GPU specific operations
    export convert_to_dev, Device

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
    include("arrays/symplectic_lie_algebra.jl")
    include("arrays/symplectic_lie_algebra_horizontal.jl")
    include("arrays/skew_symmetric.jl")
    include("arrays/stiefel_lie_algebra_horizontal.jl")
    include("arrays/grassmann_lie_algebra_horizontal.jl")
    include("arrays/auxiliary.jl")

    export SymmetricMatrix, SymplecticPotential, SkewSymMatrix
    export StiefelLieAlgHorMatrix
    export SymplecticLieAlgMatrix, SymplecticLieAlgHorMatrix
    export GrassmannLieAlgHorMatrix
    export StiefelProjection, SymplecticProjection

    include("orthogonalization_procedures/symplectic_householder.jl")

    #symplectic Householder routine 
    export sr, sr!


    export AbstractLayer
    export FeedForwardLayer, LinearFeedForwardLayer
    export Gradient
    export Linear
    export ResidualLayer
    export LinearSymplecticLayerP, LinearSymplecticLayerQ
    export SymplecticStiefelLayer

    include("manifolds/abstract_manifold.jl")
    include("manifolds/stiefel_manifold.jl")
    include("manifolds/symplectic_stiefel_manifold.jl")
    include("manifolds/grassmann_manifold.jl")

    export StiefelManifold, SymplecticStiefelManifold, GrassmannManifold, Manifold
    export rgrad, metric


    include("layers/abstract_layer.jl")
    include("layers/feed_forward_layer.jl")
    include("layers/gradient.jl")
    include("layers/linear.jl")
    include("layers/resnet.jl")
    include("layers/linear_symplectic_layer.jl")
    include("layers/manifold_layer.jl")
    include("layers/stiefel_layer.jl")
    include("layers/grassmann_layer.jl")
    include("layers/multi_head_attention.jl")
    include("layers/transformer.jl")
    include("layers/psd_like_layer.jl")


    #include("layers/symplectic_stiefel_layer.jl")
    export StiefelLayer, GrassmannLayer, ManifoldLayer
    export PSDLayer
    export MultiHeadAttention
    export Transformer
    export AbstractNeuralNetwork

    #INCLUDE OPTIMIZERS
    export AbstractMethodOptimiser, AbstractCache
    export GradientOptimizer, GradientCache
    export MomentumOptimizer, MomentumCache
    export AdamOptimizer, AdamCache

    export Optimizer
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
\
    #INCLUDE BACKENDS
    export AbstractNeuralNetwork
    export LuxBackend
    export NeuralNetwork

    include("architectures/architectures.jl")
    include("backends/backends.jl")
    include("backends/lux.jl")

    # set default backend in NeuralNetwork constructor
    NeuralNetwork(arch::AbstractArchitecture; kwargs...) = NeuralNetwork(arch, LuxBackend(); kwargs...)

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



    include("rng/random_funcs.jl")

    
end
