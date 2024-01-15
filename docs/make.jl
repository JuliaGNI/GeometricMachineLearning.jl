using GeometricMachineLearning
using Documenter
using DocumenterCitations
# using Weave

# this is necessary to avoid warnings. See https://documenter.juliadocs.org/dev/man/syntax/
ENV["GKSwstype"] = "100"

bib = CitationBibliography(joinpath(@__DIR__, "src", "GeometricMachineLearning.bib"))

# if the docs are generated with github actions, then this changes the path; see: https://github.com/JuliaDocs/Documenter.jl/issues/921 
const buildpath = haskey(ENV, "CI") ? ".." : ""

makedocs(;
    plugins=[bib],
    modules=[GeometricMachineLearning],
    authors="Michael Kraus, Benedikt Brantner",
    repo="https://github.com/JuliaGNI/GeometricMachineLearning.jl/blob/{commit}{path}#L{line}",
    sitename="GeometricMachineLearning.jl",
    format=Documenter.HTML(;
        repolink="https://github.com/JuliaGNI/GeometricMachineLearning.jl",
        prettyurls=get(ENV, "CI", "false") == "true",
        # not sure why we need this?
        canonical="https://juliagni.github.io/GeometricMachineLearning.jl",
        assets=[
            "assets/extra_styles.css",
        ],
        # specifies that we do not display the package name again (it's already in the logo)
        sidebar_sitename=false,
    ),
    pages=[
        "Home" => "index.md",
        "Architectures" => [
            "SympNet" => "architectures/sympnet.md",
        ],
        "Manifolds" => [
            "Concepts from General Topology" => "manifolds/basic_topology.md",
            "General Theory on Manifolds" => "manifolds/manifolds.md",
            "The Inverse Function Theorem" => "manifolds/inverse_function_theorem.md",
            "The Submersion Theorem" => "manifolds/submersion_theorem.md",
            "Homogeneous Spaces" => "manifolds/homogeneous_spaces.md",
            "Stiefel" => "manifolds/stiefel_manifold.md",
            "Grassmann" => "manifolds/grassmann_manifold.md",
            "Differential Equations and the EAU theorem" => "manifolds/existence_and_uniqueness_theorem.md",
            ],
        "Arrays" => [
            "Global Tangent Space" => "arrays/stiefel_lie_alg_horizontal.md",
        ],
        "Optimizer Framework" => [
            "Optimizers" => "Optimizer.md",
            "General Optimization" => "optimizers/general_optimization.md",
        ],
        "Optimizer Functions" => [
            "Horizontal Lift" => "optimizers/manifold_related/horizontal_lift.md",
            "Global Sections" => "optimizers/manifold_related/global_sections.md",
            "Retractions" => "optimizers/manifold_related/retractions.md",
            "Geodesic Retraction" => "optimizers/manifold_related/geodesic.md",
            "Cayley Retraction" => "optimizers/manifold_related/cayley.md",
            "Adam Optimizer" => "optimizers/adam_optimizer.md",
            "BFGS Optimizer" => "optimizers/bfgs_optimizer.md",
            ],
        "Special Neural Network Layers" => [
            "Attention" => "layers/attention_layer.md",
            "Multihead Attention" => "layers/multihead_attention_layer.md",
        ],
        "Data Loader" =>[
            "Routines" => "data_loader/data_loader.md",
            "Snapshot matrix" => "data_loader/snapshot_matrix.md",
        ],
        "Reduced Order Modelling" =>[
            "POD and Autoencoders" => "reduced_order_modeling/autoencoder.md",
            "PSD and Symplectic Autoencoders" => "reduced_order_modeling/symplectic_autoencoder.md",
            "Kolmogorov n-width" => "reduced_order_modeling/kolmogorov_n_width.md",
            "Projection and Reduction Error" => "reduced_order_modeling/projection_reduction_errors.md",
        ],
        "Tutorials" =>[
            "Sympnets" => "tutorials/sympnet_tutorial.md",
            "Linear Wave Equation" => "tutorials/linear_wave_equation.md",
            "MNIST" => "tutorials/mnist_tutorial.md",
            "Grassmann manifold" => "tutorials/grassmann_layer.md",
        ],
        "References" => "references.md",
        "Library" => "library.md",
    ],
)

deploydocs(;
    repo   = "github.com/JuliaGNI/GeometricMachineLearning.jl",
    devurl = "latest",
    devbranch = "main",
)
