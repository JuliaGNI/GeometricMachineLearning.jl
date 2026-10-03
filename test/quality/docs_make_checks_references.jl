# `docs/make.jl` runs `docs/check_references.jl` before `makedocs`, so that a bad `@ref` stops the
# build in the time it takes to load the package rather than after the tutorials have run.
#
# This replaces the docstring of `PositionalEncoding`, which `docs/src/layers/positional_encoding.md`
# includes, by one with a `@ref` to a binding that does not exist, and runs `make.jl` in a fresh
# process. The test environment holds what `check_references.jl` loads but not the packages that
# `make.jl` loads after it (DocumenterCitations among them), so a build that gets past the check
# fails on a missing package, not with the check's error.

using Test

const MAKE = joinpath(dirname(dirname(@__DIR__)), "docs", "make.jl")
const BAD = "no_binding_of_this_name_exists"

script = """
import GeometricMachineLearning
Core.eval(GeometricMachineLearning, :(@doc "See [`$(BAD)`](@ref)." PositionalEncoding))
include($(repr(MAKE)))
"""

cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`
output = IOBuffer()
process = run(pipeline(ignorestatus(cmd); stdout = output, stderr = output))
text = String(take!(output))

@test !success(process)
# The check reports exactly the one reference this test broke, so the page set and every other
# reference of the manual resolve in this environment.
@test occursin("FAIL <docstring of PositionalEncoding>", text)
@test occursin(BAD, text)
@test occursin("\n1 unresolved reference(s)", text)
# The process stops on the check's own error, not on anything `make.jl` does after it.
@test occursin(r"ERROR: (LoadError: )*1 unresolved reference\(s\)", text)
@test !occursin("DocumenterCitations", text)
