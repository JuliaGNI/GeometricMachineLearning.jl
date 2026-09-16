using SafeTestsets, Test, GeometricMachineLearning

# A test that trains passes `show_progress = false`. The `Optimizer` functor defaults it to `true`,
# which is right at a REPL and is noise in a suite -- a 2048-epoch run emits a few hundred progress
# lines and buries the failure you are looking for.

# One directory per subject, each with its own driver naming the testsets it runs. The drivers are
# `include`d at top level rather than wrapped in a testset of their own, because `@safetestset`
# expands to a `module` and a module may not appear inside a testset body.
#
# The two guards stay here beside `runtests.jl` rather than in a directory of their own: they check
# the tree rather than a subject in it, and `reachability.jl` reads `test/` off its own `@__DIR__`.

@safetestset "Reachability of every file under test/" begin
    include("reachability.jl")
end
@safetestset "Exported names are defined" begin
    include("exports.jl")
end

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
