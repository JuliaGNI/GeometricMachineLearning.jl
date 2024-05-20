using GeometricMachineLearning
using Documenter
using DocumenterCitations
using Markdown
using Bibliography
# using Weave

# this is necessary to avoid warnings. See https://documenter.juliadocs.org/dev/man/syntax/
ENV["GKSwstype"] = "100"

bib = CitationBibliography(joinpath(@__DIR__, "src", "GeometricMachineLearning.bib"))
sort_bibliography!(bib.entries, :nyt)  # name-year-title

# if the docs are generated with github actions, then this changes the path; see: https://github.com/JuliaDocs/Documenter.jl/issues/921 
const buildpath = haskey(ENV, "CI") ? ".." : ""

const html_format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", nothing) == "true",
    repolink = "https://github.com/JuliaGNI/GeometricMachineLearning.jl",
    canonical = "https://juliagni.github.io/GeometricMachineLearning.jl",
    assets = [
        "assets/extra_styles.css",
        ],
    # specifies that we do not display the package name again (it's already in the logo)
    sidebar_sitename = false,
    )

# if platform is set to "none" then no output pdf is generated
const latex_format = Documenter.LaTeX(platform = "none")

# output_type is defined here for handling e.g. figures in a not-too-messy way 
# if we supply no arguments to make.jl or supply html_output, then `output_type` is `:html`. Else it is latex.
const output_type = isempty(ARGS) ? :html : ARGS[1] == "html_output" ? :html : :latex

# the format is needed by the Julia documenter
const format = output_type == :html ? html_format : latex_format

function html_graphics(path::String; kwargs...)
    light_path = joinpath(path * ".png")
    dark_path = joinpath(path * "_dark.png")
    light_string = """<object type="image/svg+xml" class="display-light-only" data=$(joinpath(buildpath, light_path))></object>"""
    dark_string = """<object type="image/svg+xml" class="display-dark-only" data=$(joinpath(buildpath, dark_path))></object>"""
    @assert isfile(light_path) "No file found for " * light_path * "!"
    @assert isfile(dark_path) "No file found for " * dark_path * "!"
    Docs.HTML(light_string, dark_string)
end

function latex_graphics(path::String; label = nothing, caption = nothing, width = .5)
    figure_width = "$(width)\\textwidth"
    latex_label = isnothing(label) ? "" : "\\label{" * label * "}" 
    latex_caption = isnothing(caption) ? "" : "\\caption{" * caption * "}"
    latex_string = """\\begin{figure}
            \\includegraphics[width = """ * figure_width * "]{" * path * ".png}" *
            latex_caption *
            latex_label * """
        \\end{figure}"""
end

function include_graphics(path::String; kwargs...)
    Main.output_type == :html ? html_graphics(path; kwargs...) : latex_graphics(path; kwargs...)
end

function theorem(statement::String, name::Nothing; label::Union{Nothing, String} = nothing)
    if Main.output_type == :html
        Markdown.parse("__Theorem:__ *" * statement * "*")
    else
        theorem_label = isnothing(label) ? "" : raw"\label{th:" * label * raw"}"
        Markdown.parse(raw"\begin{thrm}" * statement * theorem_label * raw"\end{thrm}")
    end
end

function theorem(statement::String, name::String; label::Union{Nothing, String} = nothing)
    if Main.output_type == :html
        Markdown.parse("__Theorem (" * name * "):__ *" * statement * "*")
    else
        theorem_label = isnothing(label) ? "" : raw"\label{th:" * label * raw"}"
        Markdown.parse(raw"\begin{thrm}[" * name * "]" * statement * theorem_label * raw"\end{thrm}")
    end
end

function theorem(statement::String; name::Union{Nothing, String} = nothing, label::Union{Nothing, String} = nothing)
    theorem(statement, name; label = label)
end

function definition(statement::String; label::Union{Nothing, String} = nothing)
    if Main.output_type == :html
        Markdown.parse("__Definition:__ *" * statement * "*")
    else
        theorem_label = isnothing(label) ? "" : raw"\label{def:" * label * raw"}"
        Markdown.parse(raw"\begin{dfntn}" * statement * theorem_label * raw"\end{dfntn}")
    end
end

function example(statement::String; label::Union{Nothing, String} = nothing)
    if Main.output_type == :html
        Markdown.parse("__Example:__ *" * statement * "*")
    else
        theorem_label = isnothing(label) ? "" : raw"\label{xmpl:" * label * raw"}"
        Markdown.parse(raw"\begin{xmpl}" * statement * theorem_label * raw"\end{xmpl}")
    end
end

makedocs(;
    plugins = [bib],
    modules = [GeometricMachineLearning],
    authors = "Michael Kraus, Benedikt Brantner",
    repo = "https://github.com/JuliaGNI/GeometricMachineLearning.jl/blob/{commit}{path}#L{line}",
    sitename = "GeometricMachineLearning.jl",
    format = format,
    pages=[
        "Home" => "index.md",
        "Manifolds" => [
            "Concepts from General Topology" => "manifolds/basic_topology.md",
            "Foundations of Differential Manifolds" => "manifolds/inverse_function_theorem.md",
            "General Theory on Manifolds" => "manifolds/manifolds.md",
            "Homogeneous Spaces" => "manifolds/homogeneous_spaces.md",
            "Stiefel" => "manifolds/stiefel_manifold.md",
            "Grassmann" => "manifolds/grassmann_manifold.md",
            "Differential Equations and the EAU theorem" => "manifolds/existence_and_uniqueness_theorem.md",
            ],
        "Arrays" => [
            "Symmetric and Skew-Symmetric Matrices" => "arrays/skew_symmetric_matrix.md",
            "Stiefel Global Tangent Space" => "arrays/stiefel_lie_alg_horizontal.md",
            "Grassmann Global Tangent Space"=> "arrays/grassmann_lie_alg_hor_matrix.md",
        ],
        "Optimizer Framework" => [
            "Optimizers" => "Optimizer.md",
            "General Optimization" => "optimizers/general_optimization.md",
            "Pullbacks" => "pullbacks/computation_of_pullbacks.md",
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
            "Sympnet Gradient Layers" => "layers/sympnet_gradient.md",
            "Volume-Preserving Layers" => "layers/volume_preserving_feedforward.md",
            "Attention" => "layers/attention_layer.md",
            "Multihead Attention" => "layers/multihead_attention_layer.md",
            "Linear Symplectic Attention" => "layers/linear_symplectic_attention.md",
        ],
        "Architectures" => [
            "Symplectic Autoencoders" => "architectures/symplectic_autoencoder.md",
            "Neural Network Integrators" => "architectures/neural_network_integrators.md",
            "SympNet" => "architectures/sympnet.md",
            "Volume-Preserving FeedForward" => "architectures/volume_preserving_feedforward.md",
            "Standard Transformer" => "architectures/transformer.md",
            "Volume-Preserving Transformer" => "architectures/volume_preserving_transformer.md",
            "Linear Symplectic Transformer" => "architectures/linear_symplectic_transformer.md",
        ],
        "Data Loader" =>[
            "Routines" => "data_loader/data_loader.md",
            "Snapshot matrix & tensor" => "data_loader/snapshot_matrix.md",
        ],
        "Reduced Order Modelling" =>[
            "POD and Autoencoders" => "reduced_order_modeling/autoencoder.md",
            "PSD and Symplectic Autoencoders" => "reduced_order_modeling/symplectic_autoencoder.md",
            "Kolmogorov n-width" => "reduced_order_modeling/kolmogorov_n_width.md",
            "Projection and Reduction Error" => "reduced_order_modeling/projection_reduction_errors.md",
        ],
        "Tutorials" =>[
            "Sympnets" => "tutorials/sympnet_tutorial.md",
            "Symplectic Autoencoders" => "tutorials/symplectic_autoencoder.md",
            "MNIST" => "tutorials/mnist_tutorial.md",
            "Grassmann manifold" => "tutorials/grassmann_layer.md",
            "Volume-Preserving Attention" => "tutorials/volume_preserving_attention.md",
            "Linear Symplectic Transformer" => "tutorials/linear_symplectic_transformer.md",
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
