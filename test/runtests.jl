using ParallelTestRunner
using GeometricMachineLearning

const TESTS = [
    "core" => [
        "MatrixSoftmax" => "activations/softmax.jl",
        "Test triangular matrices" => "integration/triangular.jl",
        "_mul: left, right and tensor products" => "kernels/mul.jl",
        "assign_q_and_p splits by slicing" => "kernels/assign_q_and_p.jl",
        "Custom AD rules for kernels" => "kernels/kernel_pullbacks.jl",
        "The products of src/kernels/ on a JLArray" => "devices/jlarray.jl",
        "train! on a JLArray: the Dense and the StiefelLayer chains" => "devices/jlarray_layers.jl",
        "train! on a JLArray: the three symplectic networks" => "devices/jlarray_sympnet.jl",
        "train! on a JLArray: the symplectic autoencoder" => "devices/jlarray_autoencoder.jl",
        "train! on a JLArray: the standard transformer integrator" => "devices/jlarray_transformer.jl",
        "A pullback of a device network returns its gradient leaves on the device" => "devices/jlarray_gradient_backend.jl",
        "Product gradients of structured and Stiefel weights" => "integration/product_gradients.jl",
        "A Stiefel network trains on data of the other precision" => "integration/mixed_precision_training.jl",
        "tensor_cayley: skew inputs, the orthogonality bound and the pullback" => "kernels/cayley.jl",
        "Volume-Preserving Transformer (skew-symmetric tests)" => "kernels/test_skew_map.jl",
        "Documented keyword defaults are the constructed ones" => "integration/documented_defaults.jl",
        "_norm keeps the element type of its argument" => "utils.jl",
        "ZygotePullback returns a NetworkParameters" => "pullbacks/zygote_pullback.jl",
        "AutoEncoderLoss and a loss's NetworkParameters" => "loss/losses.jl",
        "apply_parametric and the concatenation of the system parameters" => "parametric.jl",
        "ParametricLoss and its ZygotePullback" => "loss/parametric_loss.jl",
        "map_to_cpu" => "map_to_cpu.jl",
        "changebackend" => "integration/changebackend_tests.jl",
        "HDF5 save/load for GML special array types" => "integration/hdf5_support.jl",
        "Layer and architecture docstring examples" => "integration/docstrings/layers_and_architectures.jl",
        "Loss docstring examples" => "integration/docstrings/losses.jl",
        "Manifold docstring examples" => "integration/docstrings/manifolds.jl",
        "Utility and pullback docstring examples" => "integration/docstrings/utilities.jl",
        "The (q, p) methods dispatch on NamedTupleOfArrays" => "integration/qp_dispatch.jl"
    ],
    "quality" => [
        "Reachability of every file under test/" => "quality/reachability.jl",
        "Exported names are defined" => "quality/exports.jl",
        "Aqua's package-level checks" => "quality/aqua.jl",
        "JET" => "quality/jet.jl",
        "docs/make.jl stops on a bad @ref before makedocs" => "quality/docs_make_checks_references.jl"
    ],
    "layers" => [
        "Gradient Layer" => "layers/gradient_layer_tests.jl",
        "Analytic inverse layers" => "layers/inverse_layers.jl",
        "Permutation-symmetric particle layers" => "layers/symmetric_particle_layers.jl",
        "Bilinear and low-rank Stiefel layers" => "layers/bilinear_and_low_rank_stiefel.jl",
        "Forcing layers add to p and leave q" => "layers/forcing_layer.jl",
        "Test tensor-slice consistency of sympnet layers" => "layers/sympnet_layers_test.jl",
        "Manifold Neural Network Layers" => "layers/manifold_layers.jl",
        "The manifold layers initialise an orthonormal weight" => "layers/manifold_layer_orthonormality.jl",
        "ResNet" => "layers/resnet_tests.jl",
        "WideResNetLayer" => "layers/wide_resnet.jl",
        "ParametricResNetLayer" => "layers/parametric_resnet_layer.jl",
        "Classification layer" => "layers/classification.jl",
        "Test volume-preserving feedforward neural network" => "layers/volume_preserving_feedforward.jl",
        "parameterlength(::PSDLayer) is exact" => "layers/psd_parameterlength.jl",
        "parameterlength(::GrassmannLayer) is an integer count" => "layers/grassmann_parameterlength.jl",
        "Attention layer #1" => "layers/attention_setup.jl",
        "VolumePreservingAttention at every sequence length" => "layers/volume_preserving_attention.jl",
        "Test setup of MultiHeadAttention layer Stiefel weights" => "layers/multi_head_attention_stiefel_setup.jl",
        "Test geodesic and Cayley retr for the MultiHeadAttention layer w/ St weights" => "layers/multi_head_attention_stiefel_retraction.jl",
        "Test the correct setup of the various optimizer caches for MultiHeadAttention" => "layers/multi_head_attention_stiefel_optim_cache.jl",
        "Linear Symplectic Attention" => "layers/linear_symplectic_attention.jl",
        "parameterlength(::MultiHeadAttention{M,M,true}) is exact" => "layers/multi_head_attention_parameterlength.jl",
        "compute_output_of_mha infers concretely" => "layers/multi_head_attention_inference.jl",
        "MultiHeadAttention builds no head key per forward pass" => "layers/multi_head_attention_allocation.jl",
        "Layer forward passes against the recorded outputs" => "layers/layer_pins.jl",
        "Layer type parameters, shared methods and type arguments" => "layers/layer_structure.jl",
        "Test setup of transformer with Stiefel weights" => "layers/transformer_setup.jl",
        "Sinusoidal positional encoding" => "layers/positional_encoding.jl",
        "Check if the transformer can be applied to a tensor." => "layers/transformer_application.jl",
        "Check if the gradient/pullback of MultiHeadAttention changes type in St case" => "layers/transformer_gradient.jl",
        "Check if the optimization_step! changes the parameters of the transformer" => "layers/transformer_optimizer.jl",
        "_mul: NetworkParameters gradient structure" => "layers/double_multiplication_network_parameters_gradient.jl",
        "Symplectic attention: NetworkParameters gradient structure" => "layers/symplectic_attention_network_parameters_gradient.jl",
        "Symplectic attention: single sample and symplecticity" => "layers/symplectic_attention.jl"
    ],
    "architectures" => [
        "Regular transformer integrator" => "architectures/standard_transformer_integrator.jl",
        "Linear Symplectic Transformer" => "architectures/linear_symplectic_transformer.jl",
        "Symplectic Transformer chain construction" => "architectures/symplectic_transformer_chain.jl",
        "Check parameterlength" => "architectures/check_parameterlengths.jl",
        "SympNet integrator" => "architectures/sympnet_integrator.jl",
        "ForcedSympNet: layers, forcing and training" => "architectures/forced_sympnet.jl",
        "PSD tests" => "architectures/psd_architecture_tests.jl",
        "SymplecticAutoencoder tests" => "architectures/symplectic_autoencoder_tests.jl",
        "WeightTiedSymplecticAutoencoder" => "architectures/weight_tied_symplectic_autoencoder.jl",
        "MultiModalSymplecticAutoencoder and multimodal losses" => "architectures/multimodal_symplectic_autoencoder.jl",
        "Check if autoencoder error is lower than PSD error" => "architectures/sae_error_lower_than_psd_error.jl",
        "Architecture forward passes against the recorded outputs" => "architectures/architecture_pins.jl",
        "Architecture keywords and chain builders" => "architectures/architecture_structure.jl",
        "ResNet blocks by width" => "architectures/resnet.jl",
        "ParametricResNet" => "architectures/parametric_resnet.jl",
        "The forward pass of every exported architecture infers" => "architectures/forward_inference.jl"
    ],
    "reducedsystem" => [
        "Check reduced model" => "reduced_system/reduced_system.jl",
        "HRedSys reference-state shifting and VJP pullback mode" => "reduced_system/reference_shift_and_vjp.jl",
        "ReducedLoss through train!" => "integration/reduced_loss_optimization.jl"
    ],
    "training" => [
        "Test NetworkLoss + Optimizer" => "integration/losses_and_optimization.jl",
        "Optimizer #2" => "optimizers/optimization_step.jl",
        "Optimizer #3" => "optimizers/svd_optim.jl",
        "Optimizer #4" => "optimizers/psd_optim.jl",
        "Check if Adam with decay converges" => "optimizers/adam_with_learning_rate_decay.jl",
        "Gradient optimizer tests" => "optimizers/gradient_optimizer.jl",
        "Momentum optimizer tests" => "optimizers/momentum_optimizer.jl",
        "Optimizers with structured (non-manifold) weights" => "optimizers/structured_array_parameters.jl",
        "Test data loader for q and p data" => "data_loader/batch_data_loader_qp_test.jl",
        "Test the data loader in combination with optimization_step!" => "data_loader/data_loader_optimization_step.jl",
        "One epoch of train! with Adam on a data loader" => "data_loader/optimizer_functor_with_adam.jl",
        "DataLoader for input and output" => "data_loader/data_loader_for_input_and_output.jl",
        "A Float32 network's training history stays Float32" => "data_loader/training_history_eltype.jl",
        "train!: convergence, resume, history type, n_epochs" => "data_loader/train.jl",
        "train! takes a TrainingOptimizer's algorithm and linesearch" => "optimizers/optimizer.jl",
        "The Batch method of integrate forwards its prediction window" => "data_loader/batch.jl",
        "solve! of a PSD architecture needs a samples loader" => "data_loader/data_loader.jl",
        "train! on a ParametricDataLoader trains on ParametricLoss" => "data_loader/parametric_data_loader.jl"
    ],
    "slow" => [
        "Test symplecticity of the sympnet upscaling layer" => "layers/sympnet_upscaling.jl",
        "train! trains every architecture of the manual" => "integration/train_manual_architectures.jl",
        "The three-stage multi-modal pipeline and its mutants" => "integration/multimodal_pipeline.jl"
    ],
    "doctests" => [
        "Doctests" => "quality/doctests.jl"
    ],
    "metal" => [
        "Metal: the tensor products and the Cayley kernel on an Apple GPU" => "devices/metal.jl",
        "Metal: eight architectures against their host twins" => "devices/metal_check.jl"
    ]
]

const SELECTED = filter(!startswith("--"), ARGS)
const GROUPS = isempty(SELECTED) ?
               (Sys.isapple() && Sys.ARCH === :aarch64 ?
                [
    "core", "quality", "layers", "architectures", "reducedsystem", "training", "slow",
    "metal"] :
                [
    "core", "quality", "layers", "architectures", "reducedsystem", "training", "slow"]) :
               SELECTED
const UNKNOWN = setdiff(SELECTED, first.(TESTS))
isempty(UNKNOWN) ||
    error("no test group ", join(UNKNOWN, ", "), " in runtests.jl; its groups are ",
        join(first.(TESTS), ", "))

runtests(GeometricMachineLearning, filter(startswith("--"), ARGS);
    testsuite = Dict(label => :(include($(joinpath(@__DIR__, path))))
    for (group, files) in TESTS if group in GROUPS for (label, path) in files),
    serial = [label
              for (group, files) in TESTS if group in ("metal", "cuda")
              for (label, _) in files],
    serial_position = :after,
    retries = 0,
    exeflags = ["--threads=$(Threads.nthreads())"],
    max_default_jobs = max(1, Sys.CPU_THREADS ÷ Threads.nthreads()))
