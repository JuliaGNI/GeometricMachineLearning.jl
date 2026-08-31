using GeometricMachineLearning
using HDF5
using AbstractNeuralNetworks
using Documenter
using DocumenterCitations
using DocumenterInterLinks
using Markdown
using Bibliography
using LaTeXStrings
# using Weave

# The manifold, special-matrix and optimizer chapters moved to `GeometricOptimizers` along with the
# types they describe (see the changelog), and the chapters that stayed refer to them constantly.
# This is what keeps those references *references* rather than prose. Closes issue C3, which asked
# for the same thing for `𝔄`, `cayley` and `update!`.
#
# The inventory is given as a *committed file* as well as a URL. The URL is fetched and takes
# precedence; the committed copy is what makes this build work offline, and what keeps a docs build
# from failing because upstream happened to be mid-deploy. Regenerate it after an upstream docs
# release — from the *deployed* inventory of the version `[compat]` requires, pinned rather than
# `stable/`, so the command reproduces the same file after the next release:
#
#     julia --project=docs -e 'using DocInventories; DocInventories.save(
#         "docs/inventories/GeometricOptimizers.toml",
#         Inventory("https://juliagni.github.io/GeometricOptimizers.jl/v0.5.0/objects.inv";
#                   root_url = "https://juliagni.github.io/GeometricOptimizers.jl/stable/"))'
#
# `save` is not exported — it has to be qualified. `root_url` is not written to the TOML either; it
# is `InterLinks` below that supplies the root at build time.
#
links = InterLinks(
    "GeometricOptimizers" => (
    "https://juliagni.github.io/GeometricOptimizers.jl/stable/",
    joinpath(@__DIR__, "inventories", "GeometricOptimizers.toml")
),
)

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
    # we should get rid of this line again eventually. We will be able to do this once we got rid of library.md
    size_threshold = 1048576
)

# if platform is set to "none" then no output pdf is generated
const latex_format = Documenter.LaTeX(platform = "none")

# output_type is defined here for handling e.g. figures in a not-too-messy way 
# if we supply no arguments to make.jl or supply html_output, then `output_type` is `:html`. Else it is latex.
const output_type = isempty(ARGS) ? :html : ARGS[1] == "html_output" ? :html : :latex

# the format is needed by the Julia documenter
const format = output_type == :html ? html_format : latex_format

function theorem(statement::String, name::Nothing; label::Union{Nothing, String} = nothing)
    if Main.output_type == :html
        Markdown.parse("""!!! info "Theorem" 
            \t $(statement)""")
    else
        theorem_label = isnothing(label) ? "" : raw"\label{th:" * label * raw"}"
        Markdown.parse(raw"\begin{thrm}" * statement * theorem_label * raw"\end{thrm}")
    end
end

function theorem(statement::String, name::String; label::Union{Nothing, String} = nothing)
    if Main.output_type == :html
        Markdown.parse("""!!! info "Theorem ($(name))" 
            \t $(statement)""")
    else
        theorem_label = isnothing(label) ? "" : raw"\label{th:" * label * raw"}"
        Markdown.parse(raw"\begin{thrm}[" * name * "]" * statement * theorem_label *
                       raw"\end{thrm}")
    end
end

function theorem(statement::String; name::Union{Nothing, String} = nothing,
        label::Union{Nothing, String} = nothing)
    theorem(statement, name; label = label)
end

function definition(statement::String; label::Union{Nothing, String} = nothing)
    if Main.output_type == :html
        Markdown.parse("""!!! info "Definition" 
            \t $(statement)""")
    else
        theorem_label = isnothing(label) ? "" : raw"\label{def:" * label * raw"}"
        Markdown.parse(raw"\begin{dfntn}" * statement * theorem_label * raw"\end{dfntn}")
    end
end

function example(statement::String; label::Union{Nothing, String} = nothing)
    if Main.output_type == :html
        Markdown.parse("""!!! info "Example" 
            \t $(statement)""")
    else
        theorem_label = isnothing(label) ? "" : raw"\label{xmpl:" * label * raw"}"
        Markdown.parse(raw"\begin{xmpl}" * statement * theorem_label * raw"\end{xmpl}")
    end
end

function remark(statement::String; label::Union{Nothing, String} = nothing)
    if Main.output_type == :html
        Markdown.parse("""!!! tip "Remark" 
            \t $(statement)""")
    else
        theorem_label = isnothing(label) ? "" : raw"\label{rmrk:" * label * raw"}"
        Markdown.parse(raw"\begin{rmrk}" * statement * theorem_label * raw"\end{rmrk}")
    end
end

function proof(statement::String)
    if Main.output_type == :html
        Markdown.parse("""!!! details "Proof" 
            \t $(statement)""")
    else
        Markdown.parse(raw"\begin{proof}" * statement * raw"\end{proof}")
    end
end

function sphere(r, C)   # r: radius; C: center [cx,cy,cz]
    n = 100
    u = range(-π, π; length = n)
    v = range(0, π; length = n)
    x = C[1] .+ r * cos.(u) * sin.(v)'
    y = C[2] .+ r * sin.(u) * sin.(v)'
    z = C[3] .+ r * ones(n) * cos.(v)'
    x, y, z
end

# this is needed if we have multiline definitions or proofs
const indentation = output_type == :html ? "\t" : ""

_introduction_text = output_type == :html ? "index.md" : "introduction.md"

_introduction = output_type == :html ? ("HOME" => "index.md") :
                ("HOME" =>
    ["acknowledgements.md",
        "abstract.md",
        "zusammenfassung.md",
        "toc.md",
        "introduction.md"]
)

# The `Manifolds` chapter — general topology through homogeneous spaces — moved to
# `GeometricOptimizers` with the manifold types it describes, as did `Symmetric and Skew-Symmetric
# Matrices` and `Global Tangent Spaces` below and the whole `Optimizer` chapter. They are linked
# into from here through `InterLinks`.

_special_arrays = "Special Arrays and AD" => [
    "Tensors" => "arrays/tensors.md",
    "Pullbacks" => "pullbacks/computation_of_pullbacks.md"
]

_structure_preservation = "Structure-Preservation" => [
    "Symplecticity" => "structure_preservation/symplecticity.md",
    "Volume-Preservation" => "structure_preservation/volume_preservation.md",
    "Structure-Preserving Neural Networks" => "structure_preservation/structure_preserving_neural_networks.md"
]

# What is left of the optimizer chapter: the framework belongs to `GeometricOptimizers`, and this
# page covers the part that is about neural networks — the parameter tree and the training loop.
_optimizers = "Optimizer" => "optimizers/optimizer.md"

_special_layers = "Special Neural Network Layers" => [
    "Sympnet Layers" => "layers/sympnet_gradient.md",
    "Volume-Preserving Layers" => "layers/volume_preserving_feedforward.md",
    "(Volume-Preserving) Attention" => "layers/attention_layer.md",
    "Multihead Attention" => "layers/multihead_attention_layer.md",
    "Linear Symplectic Attention" => "layers/linear_symplectic_attention.md",
    "Symplectic Attention" => "layers/symplectic_attention.md"
]

_reduced_order_modeling = "Reduced Order Modeling" => [
    "General Framework" => "reduced_order_modeling/reduced_order_modeling.md",
    "POD and Autoencoders" => "reduced_order_modeling/pod_autoencoders.md",
    "Losses and Errors" => "reduced_order_modeling/losses.md",
    "Symplectic Model Order Reduction" => "reduced_order_modeling/symplectic_mor.md"
]

_architectures = "Architectures" => [
    "Using Architectures with `NeuralNetwork`" => "architectures/abstract_neural_networks.md",
    "Symplectic Autoencoders" => "architectures/symplectic_autoencoder.md",
    "Neural Network Integrators" => "architectures/neural_network_integrators.md",
    "Hamiltonian Neural Network" => "architectures/hamiltonian_neural_network.md",
    "SympNet" => "architectures/sympnet.md",
    "Volume-Preserving FeedForward" => "architectures/volume_preserving_feedforward.md",
    "Standard Transformer" => "architectures/transformer.md",
    "Volume-Preserving Transformer" => "architectures/volume_preserving_transformer.md",
    "Linear Symplectic Transformer" => "architectures/linear_symplectic_transformer.md",
    "Symplectic Transformer" => "architectures/symplectic_transformer.md"
]

_data_loader = "Data Loader" => [
    "Snapshot matrix & tensor" => "data_loader/snapshot_matrix.md",
    "Routines" => "data_loader/data_loader.md"
]

_tutorials = "Tutorials" => [
    "SympNets" => "tutorials/sympnet_tutorial.md",
    "Hamiltonian Neural Network" => "tutorials/hamiltonian_neural_network.md",
    "Symplectic Autoencoders" => "tutorials/symplectic_autoencoder.md",
    "Grassmann Manifold" => "tutorials/grassmann_layer.md",
    "Volume-Preserving Attention" => "tutorials/volume_preserving_attention.md",
    "Matrix Attention" => "tutorials/matrix_softmax.md",
    "Volume-Preserving Transformer for the Rigid Body" => "tutorials/volume_preserving_transformer_rigid_body.md",
    "Linear Symplectic Transformer" => "tutorials/linear_symplectic_transformer.md",
    "Symplectic Transformer" => "tutorials/symplectic_transformer.md",
    "Adjusting the Loss Function" => "tutorials/adjusting_the_loss_function.md",
    "Comparing Optimizers" => "tutorials/optimizer_comparison.md"
]

_outlook = "Summary and Outlook" => "outlook.md"
_references = "References" => "references.md"
_index_of_docstrings = "Index of Docstrings" => "docstring_index.md"

_html_pages = [
    _introduction,
    _special_arrays,
    _structure_preservation,
    _optimizers,
    _special_layers,
    _reduced_order_modeling,
    "port-Hamiltonian Systems" => "port_hamiltonian_systems.md",
    _architectures,
    _data_loader,
    _tutorials,
    _references,
    _index_of_docstrings
]

# Maybe you want to name "Background" → "Manifolds, Global Tangent Spaces and Geometric Structure"

const SectionType = Vector{Pair{String, String}}
const ChapterType = Pair{String, SectionType}
const LatexChapterType = Pair{String, Vector{String}}
const SingleDocumentType = Pair{String, String}
const SpecialType = Pair{String, Union{String, Vector{String}}}

function reduce_to_second_factors(pair::String)
    pair
end
# this returns a vector
function reduce_to_second_factors(pairs::Vector{String})
    pairs
end
function reduce_to_second_factors(pair::SingleDocumentType)
    pair[2]
end
# this returns a vector
function reduce_to_second_factors(list::SectionType)
    [list_item[2] for list_item in list]
end
function reduce_to_second_factors(pair::ChapterType)
    reduce_to_second_factors(pair[2])
end
function reduce_to_second_factors(pairs::Vector{ChapterType})
    second_factors = Tuple([reduce_to_second_factors(pair) for pair in pairs])
    vcat(second_factors...)
end
function reduce_to_second_factors(pair::LatexChapterType)
    vcat(pair[2]...)
end
function reduce_to_second_factors(pair::Pair{String, Any})
    reduce_to_second_factors(pair[2])
end
function reduce_to_second_factors(pairs::Vector{Any})
    vcat([reduce_to_second_factors(pair) for pair in pairs]...)
end
function reduce_to_second_factors(pairs::Vector{Pair{String, Any}})
    vcat([reduce_to_second_factors(pair) for pair in pairs]...)
end

function value_for_key(pair::Pair{String, <:Any}, key::String)
    Dict(pair)[key]
end

function value_for_key(pairs::Vector{PT}, key::String) where {PT <: Pair{String, <:Any}}
    Dict(pairs)[key]
end

function value_for_key(
        pairs::Pair{
            String, VT}, key::String) where {PT <: Pair{String, <:Any}, VT <: Vector{PT}}
    value_for_key(pairs[2], key)
end

function value_for_key(pairs::Union{VT, Pair{String, VT}},
        keys...) where {
        PT <: Pair{String, <:Any},
        VT <: Vector{PT}}
    values = Vector{String}()
    for key in keys
        push!(values, value_for_key(pairs, key))
    end
    values
end

_latex_pages = [
    _introduction,
    # The `Manifolds` chapter this part opened with, and the `Optimizers` part that followed it,
    # are `GeometricOptimizers`' documentation now. The book therefore starts from the geometric
    # structure and takes the manifold optimizers as given; see the changelog.
    "Background" => [
        "Geometric Structure" => reduce_to_second_factors(_structure_preservation),
        "Reduced Order Modeling" => reduce_to_second_factors(_reduced_order_modeling)
    ],
    # One page, but it still has to be a chapter of *pairs* like the others: `index_latex_pages`
    # below flattens these values and `docstring_index.md` builds a `Dict` from the result, so a
    # chapter contributing a bare string breaks that `Dict`.
    "Optimizer" => ["Optimizer" => reduce_to_second_factors(_optimizers)],
    "Special Neural Network Layers and Architectures" => [
        "Layers" => reduce_to_second_factors(_special_layers),
        "Architectures" => reduce_to_second_factors(_architectures)
    ],
    # we do not include the last tutorial here
    "Experiments and Applications" => [
        "Learning a Reduced Model with Symplectic Autoencoders" =>
            value_for_key(_tutorials, "Symplectic Autoencoders"),
        "Neural Networks as Symplectic Integrators" => value_for_key(_tutorials,
            "SympNets",
            "Linear Symplectic Transformer"),
        "Transformers with Structure" => value_for_key(_tutorials,
            "Volume-Preserving Transformer for the Rigid Body",
            "Volume-Preserving Attention"),
        "Learning Nonlinear Spaces" => value_for_key(_tutorials, "Grassmann Manifold")
    ],
    _outlook,
    _references,
    _index_of_docstrings,
    "Appendix" => [
        "Data Loader" => reduce_to_second_factors(_data_loader),
        "Tensors and Pullbacks" =>
            value_for_key(_special_arrays, "Tensors",
                "Pullbacks"),
        # we include the last tutorial here
        "Customizing Training" =>
            value_for_key(_tutorials, "Adjusting the Loss Function"),
        "Other Structure-Preserving Properties" => "port_hamiltonian_systems.md"
    ]
]

_keys = [page[1] for page in _latex_pages]
# don't generate docstring indices for specific chapters (introduction, conclusion, ...)
filter!(
    key -> (key ≠ "HOME") & (key ≠ "Index of Docstrings") & (key ≠ "References") &
           (key ≠ "Summary and Outlook"),
    _keys)
index_latex_pages = vcat([Dict(_latex_pages)[key] for key in _keys]...)

makedocs(;
    plugins = [bib, links],
    # `GeometricOptimizers` is deliberately *not* listed. `@docs` filters candidate docstrings by
    # the module they were written in (`d.data[:module]`), not by the module of the binding, so the
    # `geodesic`/`cayley` methods GML defines on GeometricOptimizers' functions are found from here
    # anyway. Listing GeometricOptimizers would instead (i) make `missing_docs` demand that this
    # manual document all 147 of its docstrings — `checkdocs_ignored_modules` does not help, it only
    # skips *sub*modules — and (ii) pull GeometricOptimizers' own docstrings in, whose internal
    # `@ref`s resolve in its namespace and point at bindings this manual does not document.
    # Names owned by GeometricOptimizers are referred to as plain code with a link to its manual.
    modules = [
        GeometricMachineLearning, Base.get_extension(GeometricMachineLearning, :HDF5Ext)],
    authors = "Michael Kraus, Benedikt Brantner",
    repo = "https://github.com/JuliaGNI/GeometricMachineLearning.jl/blob/{commit}{path}#L{line}",
    sitename = "GeometricMachineLearning.jl",
    format = format,
    # Doctests run in the `doctest` job of `.github/workflows/CI.yml`, pinned to one OS and one
    # Julia version, where they produce a required status check of their own. Running them here as
    # well would doctest twice per invocation, and this manual is built twice — once as HTML and
    # once as LaTeX.
    doctest = false,
    pages = output_type == :html ? _html_pages : _latex_pages
)

# The tutorials' `@setup` blocks read pre-trained parameters from `docs/src/tutorials/*.h5` so that
# building the manual does not have to retrain the networks. Documenter copies every non-Markdown
# file under `src/` into the output, so those weights were published along with the manual — ~3.8 MB
# per release, and `gh-pages` accumulates a copy for every version. No page links to them (the
# `load` calls all live in hidden `@setup` blocks), so they are build *inputs* only. Drop them from
# `build/` here rather than earlier: `makedocs` needs them on disk while it evaluates the blocks.
const _build_only_data_extensions = (".h5", ".jld2")

for (root, _, files) in walkdir(joinpath(@__DIR__, "build"))
    for file in files
        any(ext -> endswith(file, ext), _build_only_data_extensions) || continue
        rm(joinpath(root, file))
    end
end

deploydocs(;
    repo = "github.com/JuliaGNI/GeometricMachineLearning.jl",
    devurl = "latest",
    devbranch = "main"
)
