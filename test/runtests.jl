using SafeTestsets, Test, GeometricMachineLearning

# A test that trains passes `show_progress = false`. The `Optimizer` functor defaults it to `true`,
# which is right at a REPL and is noise in a suite -- a 2048-epoch run emits a few hundred progress
# lines and buries the failure you are looking for.

# One directory per subject, each with its own driver naming the testsets it runs. The drivers are
# `include`d at top level rather than wrapped in a testset of their own, because `@safetestset`
# expands to a `module` and a module may not appear inside a testset body.
#
# The tree-level checks stay here beside `runtests.jl` rather than in a directory of their own:
# they check the tree rather than a subject in it, and `reachability.jl` reads `test/` off its own
# `@__DIR__`.
#
# `aqua.jl` runs after the subject drivers and not beside the other two. A top-level testset throws
# when it closes on a failure, which ends the file, so whatever runs first can hide everything
# behind it. `reachability.jl` and `exports.jl` earn that position: they say whether the suite is
# well-formed at all, and a failure in either makes the rest not worth reading. Aqua's checks do
# not -- a package-level finding says nothing about whether a subject is correct, and placed first
# it costs every subject result in the run.

@safetestset "Reachability of every file under test/" begin
    include("reachability.jl")
end
@safetestset "Exported names are defined" begin
    include("exports.jl")
end

include("activations/runtests.jl")
include("arrays/runtests.jl")
include("kernels/runtests.jl")
include("layers/runtests.jl")
include("attention/runtests.jl")
include("transformers/runtests.jl")
include("architectures/runtests.jl")
include("reduced_order_modeling/runtests.jl")
include("losses/runtests.jl")
include("optimizers/runtests.jl")
include("parameters/runtests.jl")
include("data_loader/runtests.jl")
include("docstrings/runtests.jl")

@safetestset "Aqua's package-level checks" begin
    include("aqua.jl")
end
