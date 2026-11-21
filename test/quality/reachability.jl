using Test

# A test file under `test/` that nothing includes is never run, and nothing says so: it keeps
# compiling in a reader's head, it is repaired by hand from time to time, and it regresses in
# silence. This guard walks the transitive `include` closure of `runtests.jl` and of each path of
# its table `TESTS`, which ParallelTestRunner includes by that path in a worker, and requires every
# `test/**/*.jl` to be either inside that closure or named in `ALLOWED_ORPHANS` below with a
# reason. Adding a test file without wiring it in therefore fails the suite, and a file that is
# deliberately not run has to say why in one line. The allowlist is closed in both directions: an
# entry whose file is later wired in or deleted fails until the entry goes with it.
#
# The closure is read off the parsed syntax tree rather than off the text. A scanner over lines has
# to re-implement Julia's lexer to know whether an `include` it found is code, and gets `#= … =#`
# and string literals wrong; the parser has already made that distinction.
#
# The allowlist is empty, so every file under `test/` is reached from `runtests.jl`. An entry is a
# backlog item, not a design: it names a file waiting to be repaired, wired in or deleted, and its
# reason is the observed first blocker rather than a guess from the file's name.

const ALLOWED_ORPHANS = Dict{String, String}()

const TEST_ROOT = normpath(joinpath(@__DIR__, ".."))

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
Return the paths of the table `const TESTS = [...]` at the top level of `entry`, relative to
`test/`. A file without that table, such as one in the serial form, gives none.

The table is the literal `["<group>" => ["<label>" => "<path>", ...], ...]` that the layout check
of the convention requires, so it is read by position: a table of another shape raises an error
here rather than reaching fewer files.
"""
function tests_table_paths(entry::AbstractString)
    file = joinpath(TEST_ROOT, entry)
    ast = Meta.parseall(read(file, String); filename = file)
    paths = String[]
    for ex in ast.args
        Meta.isexpr(ex, :const) && ex.args[1].args[1] === :TESTS || continue
        for group in ex.args[1].args[2].args
            for pair in group.args[3].args
                push!(paths, pair.args[3])
            end
        end
    end
    return paths
end

"""
Return the transitive `include` closure of the files `entries`, as paths relative to `test/`.
Files included from outside `test/` are dropped.
"""
function include_closure(entries::AbstractVector{<:AbstractString})
    reached = Set{String}()
    stack = [normpath(joinpath(TEST_ROOT, entry)) for entry in entries]
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

const REACHABLE = include_closure(["runtests.jl"; tests_table_paths("runtests.jl")])

@testset "every test file is reachable from runtests.jl or allowlisted" begin
    files = test_files()

    # A file may be outside the closure only if the allowlist above gives a reason for it.
    unreachable = filter(f -> !(f in REACHABLE) && !haskey(ALLOWED_ORPHANS, f), files)
    @test unreachable == String[]

    # And the allowlist may not outlive what it excuses. An entry whose file is now included, or no
    # longer exists, has to go; otherwise the next reader trusts a reason that no longer holds.
    stale = filter(f -> f in REACHABLE || !(f in files), sort!(collect(keys(ALLOWED_ORPHANS))))
    @test stale == String[]
end
