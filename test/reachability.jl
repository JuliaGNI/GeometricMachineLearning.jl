using Test

# A test file under `test/` that nothing includes is never run, and nothing says so: it keeps
# compiling in a reader's head, it is repaired by hand from time to time, and it regresses in
# silence. This guard walks the transitive `include` closure of `runtests.jl` and requires every
# `test/**/*.jl` to be either inside that closure or named in `ALLOWED_ORPHANS` below with a
# reason. Adding a test file without wiring it in therefore fails the suite, and a file that is
# deliberately not run has to say why in one line.
#
# The closure is read off the parsed syntax tree rather than off the text. A scanner over lines has
# to re-implement Julia's lexer to know whether an `include` it found is code, and gets `#= … =#`
# and string literals wrong; the parser has already made that distinction.
#
# Every reason below states what running that file in the test environment actually does. The
# groups are the observed first blocker, not a guess from the file's name: a file that is said to
# work was seen to exit cleanly, and a file that is said to be broken names the error that stops
# it. The allowlist is a backlog, not a design -- each entry is a file waiting to be repaired,
# wired in or deleted.

const ALLOWED_ORPHANS = Dict(
    # Run and assert something today. Nothing includes them, so nothing notices when that stops
    # being true.
    "custom_ad_rules/matrix_vector_multiplication.jl" => "runs `test_rrule`; a ChainRulesCore example on a local type, not on package code",
    "transformer_related/transformer_setup.jl" => "runs one `@test` on transformer setup with Stiefel weights",

    # Stop at `using` a package that is neither a dependency of this package nor a test target, so the
    # test environment cannot load them at all.
    "symplectic_autoencoders/linear_wave_equation.jl" => "reaches into `scripts/` for `assemble_matrix.jl`, which stops at `using OffsetArrays`",

    # Name something the package does not provide. Each reason is the error that actually stops the
    # file, which is not always the name a reader would predict from the file's title.
    "attention_layer/apply_multi_head_attention.jl" => "`UndefVarError: Attention`; the package provides no `Attention`",
    "integrator/test_integrator.jl" => "`MethodError: HamiltonianArchitecture(::Int64)`; the constructor takes different arguments",
    "layers/sympnet_upscaling.jl" => "`UndefVarError: GradientQ`; the layer is named `GradientLayerQ`",
    "orthogonalization_procedures/global_symplectic_section.jl" => "`UndefVarError: SymplecticStiefelManifold`; the package defines no such type",
    "orthogonalization_procedures/householder.jl" => "includes `src/optimizers/householder.jl`, which does not exist",
    "orthogonalization_procedures/symplectic_householder.jl" => "`UndefVarError: Rfac` inside `GeometricMachineLearning`",
    "orthogonalization_procedures/symplectic_householder_aux.jl" => "`MethodError: PoissonTensor(::Type{Float32}, ::Int64)`",
    "train!/test_method.jl" => "`MethodError: HamiltonianArchitecture(::Int64)`",
    "train!/test_neuralnet_solution.jl" => "`UndefVarError: timestep` inside `GeometricMachineLearning`",
    "train!/test_timer.jl" => "`MethodError: GSympNet(::Int64; nhidden)`",
    "train!/test_training.jl" => "`UndefVarError: timestep` inside `GeometricMachineLearning`, raised inside its testset",
    "train!/test_trainingSet.jl" => "`MethodError: HamiltonianArchitecture(::Int64; nhidden, width)`",
    "training_phnn.jl" => "`MethodError` passing `default_parameters`; `GeometricProblems.default_parameters` is a function, not the value the call expects",

    # Load without error and assert nothing, so "it runs" says nothing about whether it still works.
    "orthogonalization_procedures/gram_schmidt.jl" => "defines `gram_schmidt_test` and `sympl_gram_schmidt_test` and calls neither, so nothing executes"
)

const TEST_ROOT = @__DIR__

"""
Collect into `targets` every path that `ex` includes with a string literal, resolved against `dir`.
"""
function collect_includes!(targets::Vector{String}, ex, dir::AbstractString)
    ex isa Expr || return targets
    if ex.head === :call && length(ex.args) == 2 && ex.args[1] === :include &&
       ex.args[2] isa AbstractString
        push!(targets, normpath(joinpath(dir, ex.args[2])))
    end
    for arg in ex.args
        collect_includes!(targets, arg, dir)
    end
    return targets
end

"""
Return the absolute paths that `file` includes with a string literal.
"""
function included_files(file::AbstractString)
    ast = Meta.parseall(read(file, String); filename = file)
    return collect_includes!(String[], ast, dirname(file))
end

"""
Return the transitive `include` closure of `entry`, as paths relative to `test/`.
Files included from outside `test/` are dropped.
"""
function include_closure(entry::AbstractString)
    reached = Set{String}()
    stack = [normpath(joinpath(TEST_ROOT, entry))]
    while !isempty(stack)
        file = pop!(stack)
        (file in reached || !isfile(file)) && continue
        push!(reached, file)
        append!(stack, included_files(file))
    end
    relative = (replace(relpath(f, TEST_ROOT), '\\' => '/') for f in reached)
    return Set(f for f in relative if !startswith(f, ".."))
end

"""
Return every `.jl` file under `test/`, as paths relative to `test/`.
"""
function test_files()
    files = String[]
    for (root, _, names) in walkdir(TEST_ROOT)
        for name in names
            endswith(name, ".jl") &&
                push!(files, replace(relpath(joinpath(root, name), TEST_ROOT), '\\' => '/'))
        end
    end
    return sort!(files)
end

const REACHABLE = include_closure("runtests.jl")

@testset "every test file is reachable from runtests.jl or allowlisted" begin
    unreachable = filter(
        f -> !(f in REACHABLE) && !haskey(ALLOWED_ORPHANS, f), test_files())
    @test unreachable == String[]
end
