# Parse the test layout of a Julia repository, and check it against the test convention.
#
#   julia --startup-file=no test-layout.jl --check <repository> ...
#
# `--check` prints one line per violation, `<repository>: [<rule>] <what>`, and exits 1 if there
# is any; a repository in form prints nothing and exits 0. The rule tags are the decisions D1–D10
# of the convention, and `using`, `seed`, `label`, `compat` and `ci-groups` for the rules that have
# no number.
#
# The rule `ci-groups` judges test/ci-groups.toml, the optional list of the groups that CI runs as
# one job each (`ci_groups`). Its groups are the groups that an empty ARGS runs, in place of
# `core` and `slow`: GROUPS lists them, each has an `if` in runtests.jl, and they may be names
# outside the convention's.
#
# Two rules are not checked here, because no layout shows them: D6, the 60 s budget of a `core`
# file, cold, and the JET half of D4, `test/quality/jet.jl` where the package has a hot or kernel
# path. `run-tests.jl <repository> core` reports D6.
#
# The rule `compat` keeps the bounds in one place: the [compat] table of test/Project.toml or
# docs/Project.toml has no entry for a dependency that Project.toml has in [deps] or [weakdeps],
# whatever its value. Those environments contain the package, so the resolver applies the root's
# bounds; an entry there can only repeat or narrow them. A test-only or docs-only dependency, and a
# root [extras] entry, are not checked.
#
# A top-level directory of `test/` with a `Project.toml` in its tree, and no file that
# `runtests.jl` lists, is a separate suite with its own environment and runner, and its files are
# not checked (`separate_suites`).
#
# `layout(repo)` is the parser. `test-census.jl` and `run-tests.jl` read the same function.

using TOML

"One `include` of a test file, as `runtests.jl` reaches it."
struct Entry
    path::String                  # absolute, normalised
    kind::Symbol                  # :safetestset, :testset or :bare
    label::Any                    # the testset label as parsed, or nothing
    group::Union{String, Nothing} # the `if "<group>" in GROUPS` around it
    file::String                  # the runtests.jl that includes it
    line::Int
end

const GROUP_NAMES = ("core", "slow", "doctests", "metal", "cuda", "broken")
# the groups that an empty ARGS runs where test/ci-groups.toml does not list them
const DEFAULT_GROUPS = ["core", "slow"]
# the groups that run outside the CI test jobs, so a group file names none of them
const OWN_JOB_GROUPS = ("doctests", "metal", "cuda", "broken")
# a group of test/ci-groups.toml, as it ends a job name
const CI_GROUP_NAME = r"^[a-z][a-z0-9]*$"
# an extra workflow of test/ci-groups.toml, and the installed ones it may not name, in lower case
# because APFS finds `ci.yml` as CI.yml
const CI_WORKFLOW_NAME = r"^[A-Za-z0-9_-]+\.yml$"
const INSTALLED_WORKFLOWS = ("ci.yml", "documenter.yml", "tagbot.yml")
const GATE = r"\b(ARGS|ENV)\b"
const ISSUE = r"#\d+|issues/\d+"
# the top-level directories of test/ that mirror no directory of src/
const OWN_DIRS = ("quality", "helpers", "integration", "verification", "devices")

function group_of(cond)
    cond isa Expr && cond.head === :call && length(cond.args) == 3 &&
    cond.args[1] === :in && cond.args[2] isa String && cond.args[3] === :GROUPS ?
    cond.args[2] : nothing
end

"The expressions of a file, or nothing if it does not parse."
function parsefile(file)
    ast = Meta.parseall(read(file, String); filename = file)
    any(a -> a isa Expr && a.head in (:error, :incomplete), ast.args) ? nothing : ast
end

"Whether the TOML text `toml` has a `[sec]` table."
has_section(toml, sec) = occursin(Regex("^[ \\t]*\\[$sec\\]", "m"), toml)

"The files under `dir` whose names end in `ext`, sorted."
function files_in(dir, ext)
    isdir(dir) || return String[]
    sort([normpath(joinpath(d, f)) for (d, _, fs) in walkdir(dir)
          for f in fs if endswith(f, ext)])
end

"Walk `runtests.jl` and every nested `runtests.jl` it includes."
function layout(repo)
    t = joinpath(repo, "test")
    acc = (entries = Entry[], nested = String[], gates = Ref(0),
        nonliteral = Tuple{String, Int}[], parse_errors = String[])
    function scan(file)
        isfile(file) || return nothing
        ast = parsefile(file)
        ast === nothing && (push!(acc.parse_errors, file); return nothing)
        line = Ref(0)
        function walk(ex, ctx, group)
            ex isa LineNumberNode && (line[] = ex.line; return)
            ex isa Expr || return
            if ex.head in (:if, :elseif)
                g = group_of(ex.args[1])
                g === nothing && occursin(GATE, string(ex.args[1])) && (acc.gates[] += 1)
                walk(ex.args[2], ctx, g === nothing ? group : g)
                length(ex.args) > 2 && walk(ex.args[3], ctx, group)
            elseif ex.head === :macrocall &&
                   string(ex.args[1]) in ("@safetestset", "@testset")
                ex.args[2] isa LineNumberNode && (line[] = ex.args[2].line)
                label = length(ex.args) >= 3 ? ex.args[3] : nothing
                kind = Symbol(string(ex.args[1])[2:end])
                foreach(a -> walk(a, (kind, label), group), ex.args[4:end])
            elseif ex.head === :call && ex.args[1] === :include && length(ex.args) == 2
                if !(ex.args[2] isa String)
                    push!(acc.nonliteral, (file, line[]))
                    return
                end
                p = normpath(joinpath(dirname(file), ex.args[2]))
                if basename(p) == "runtests.jl"
                    push!(acc.nested, p)
                    scan(p)
                else
                    kind, label = ctx === nothing ? (:bare, nothing) : ctx
                    push!(acc.entries, Entry(p, kind, label, group, file, line[]))
                end
            else
                foreach(a -> walk(a, ctx, group), ex.args)
            end
        end
        walk(ast, nothing, nothing)
        return ast
    end
    runtests = scan(joinpath(t, "runtests.jl"))
    suites = separate_suites(t, acc.entries)
    inside(f) = any(s -> startswith(f, joinpath(s, "")), suites)
    # a nested runtests.jl is a file like any other: reached through `nested`, or not listed
    files = filter(f -> f != normpath(joinpath(t, "runtests.jl")) && !inside(f), files_in(t, ".jl"))
    return (; repo, test = t, files, suites, runtests, acc.entries, acc.nested,
        gates = acc.gates[], acc.nonliteral, acc.parse_errors)
end

"""
The top-level directories of `test/` that are a separate suite: one with a `Project.toml` in its
tree, and no file that `runtests.jl` lists. Such a suite has its own environment and its own
runner, such as device tests in `test/gpu/`, one environment per vendor,
and the rules of the convention do not apply to its files. A directory with a listed file is part
of the suite of `runtests.jl`, whatever environment it holds.
"""
function separate_suites(t, entries)
    isdir(t) || return String[]
    dirs = [normpath(joinpath(t, d)) for d in readdir(t) if isdir(joinpath(t, d))]
    filter(dirs) do d
        any(((_, _, fs),) -> "Project.toml" in fs, walkdir(d)) &&
            !any(e -> startswith(e.path, joinpath(d, "")), entries)
    end
end

"""
    ci_groups(repo) -> (groups, problems)

The groups of `test/ci-groups.toml`, which CI runs as one job each, and the reasons the file is not
in form. `groups` is `nothing` where the file is absent, or where it gives no list of names.

The file is `groups = [...]`, the groups that an empty `ARGS` runs: `core`, `slow` and the
repository's own groups, each a name of lower-case letters and digits, and none of the groups that
run outside the CI test jobs. An optional `workflows = [...]` names extra workflow files of
`.github/workflows/`, such as `Metal.yml`, whose jobs are required checks too (`ci_workflows`).
"""
function ci_groups(repo)
    f = joinpath(repo, "test", "ci-groups.toml")
    isfile(f) || return (nothing, String[])
    toml = try
        TOML.parsefile(f)
    catch e
        e isa TOML.ParserError || rethrow()
        return (nothing, ["test/ci-groups.toml does not parse as TOML"])
    end
    gs = get(toml, "groups", nothing)
    gs isa Vector && !isempty(gs) && all(g -> g isa String, gs) ||
        return (nothing, ["test/ci-groups.toml has no `groups` list of names"])
    gs = Vector{String}(gs)
    problems = String[]
    for k in sort!(collect(setdiff(keys(toml), ["groups", "workflows"])))
        push!(problems,
            "test/ci-groups.toml has the key `$k`; it has `groups` and `workflows` only")
    end
    append!(problems, last(ci_workflows(repo, toml)))
    for g in unique(gs)
        occursin(CI_GROUP_NAME, g) ||
            push!(problems, "group \"$g\" of test/ci-groups.toml is not lower-case letters and digits")
        count(==(g), gs) > 1 &&
            push!(problems, "group \"$g\" is in test/ci-groups.toml more than once")
        g in OWN_JOB_GROUPS &&
            push!(problems, "group \"$g\" is in test/ci-groups.toml, and it runs outside the CI test jobs")
    end
    for g in DEFAULT_GROUPS
        g in gs || push!(problems, "test/ci-groups.toml does not list the group \"$g\"")
    end
    return (gs, problems)
end

"""
    ci_workflows(repo, toml) -> (workflows, problems)

The `workflows` of the parsed `test/ci-groups.toml` `toml` of `repo`: extra workflow files whose
jobs `harness ci-protection` requires besides those of `CI.yml`, and the reasons the list is not
in form. Each is the name of a file in `.github/workflows/` other than the installed ones.
"""
function ci_workflows(repo, toml)
    ws = get(toml, "workflows", String[])
    ws isa Vector && all(w -> w isa String, ws) ||
        return (
            String[], ["`workflows` of test/ci-groups.toml is not a list of file names"])
    problems = String[]
    for w in unique(ws)
        occursin(CI_WORKFLOW_NAME, w) && !(lowercase(w) in INSTALLED_WORKFLOWS) ||
            push!(problems,
                "workflow \"$w\" of test/ci-groups.toml is not an extra workflow file name")
        count(==(w), ws) > 1 &&
            push!(problems, "workflow \"$w\" is in test/ci-groups.toml more than once")
        isfile(joinpath(repo, ".github", "workflows", w)) ||
            push!(problems, "workflow \"$w\" of test/ci-groups.toml is not in .github/workflows/")
    end
    return (Vector{String}(ws), problems)
end

"The groups that an empty `ARGS` runs in `repo`, outside the platform groups."
default_groups(repo) = something(first(ci_groups(repo)), DEFAULT_GROUPS)

"The group names that `repo` may use: the convention's, and those of its `test/ci-groups.toml`."
group_names(repo) = union(GROUP_NAMES, default_groups(repo))

"The groups that the top-level statements of `runtests` gate with an `if`, in order."
function group_ifs(runtests)
    [group_of(s.args[1])
     for s in runtests.args
     if s isa Expr && s.head === :if && group_of(s.args[1]) !== nothing]
end

"The definitions of `GROUPS` among the statements of `runtests`, without line numbers."
function groups_defs(runtests)
    [Base.remove_linenums!(deepcopy(s))
     for s in runtests.args
     if s isa Expr && s.head === :const && s.args[1] isa Expr &&
            s.args[1].head === :(=) && s.args[1].args[1] === :GROUPS]
end

"""
    groups_violations(repo, runtests) -> Vector{Tuple{String, String}}

The violations, as `(rule, what)`, of the groups of `repo`, whose parsed `runtests.jl` is
`runtests`: the form of `test/ci-groups.toml`, a group of `runtests.jl` that is not a name of the
convention or of that file, a group of that file without an `if`, and a `GROUPS` that does not run
those groups for an empty `ARGS`. A repository with a `metal` group has the platform form of
`GROUPS`, and only it. `verify-workflows.jl` reads the same function.
"""
function groups_violations(repo, runtests)
    out = Tuple{String, String}[]
    groups, problems = ci_groups(repo)
    foreach(p -> push!(out, ("ci-groups", p)), problems)
    default = something(groups, DEFAULT_GROUPS)
    ifs = group_ifs(runtests)
    plain = :(const GROUPS = isempty(ARGS) ? $(Expr(:vect, default...)) : ARGS)
    platform = :(const GROUPS = isempty(ARGS) ?
                                (Sys.isapple() && Sys.ARCH === :aarch64 ?
                                 $(Expr(:vect, default..., "metal")) :
                                 $(Expr(:vect, default...))) :
                                ARGS)
    for def in groups_defs(runtests)
        if "metal" in ifs
            def == platform || push!(out,
                ("D5",
                    "GROUPS is not the platform form, `isempty(ARGS) ? (Sys.isapple() && Sys.ARCH === :aarch64 ? $(repr([default; "metal"])) : $(repr(default))) : ARGS`, which a repository with a `metal` group has"))
        elseif def == platform
            push!(out, ("D5",
                "GROUPS has the platform form, and the repository has no `metal` group"))
        else
            def == plain ||
                push!(out, (
                    "D5", "GROUPS is not `isempty(ARGS) ? $(repr(default)) : ARGS`"))
        end
    end
    names = union(GROUP_NAMES, default)
    for g in unique(ifs)
        g in names || push!(out, ("D5", "group \"$g\" is not one of $(join(names, ", "))"))
    end
    for g in something(groups, String[])
        g in ifs || push!(out, ("ci-groups",
            "group \"$g\" of test/ci-groups.toml has no `if` in runtests.jl"))
    end
    return out
end

"Every sub-expression of `ex` for which `pred` holds."
function collect_exprs(pred, ex, out = Any[])
    ex isa Expr || return out
    pred(ex) && push!(out, ex)
    foreach(a -> collect_exprs(pred, a, out), ex.args)
    return out
end

"A statement as `<line>: <what it is>`: the macro for a macro call, else its head."
function describe(ex, line)
    what = ex isa Expr ? (ex.head === :macrocall ? string(ex.args[1]) : string(ex.head)) :
           repr(ex)
    return "$line: $what"
end

const RANDOM_CALLS = (
    :rand, :randn, :rand!, :randn!, :randperm, :randexp, :shuffle, :shuffle!,
    :bitrand, :sprand, :randstring)
const SEED_CALLS = (:seed!, :Xoshiro, :MersenneTwister, :StableRNG)

"The name a call calls: `f` for `f(…)` and for `M.f(…)`, else nothing."
function callee(ex)
    f = ex.args[1]
    f isa Symbol && return f
    f isa Expr && f.head === :. && f.args[end] isa QuoteNode && return f.args[end].value
    return nothing
end

"Whether `ex` is a broadcast call `f.(…)`."
function is_broadcast(ex)
    ex.head === :. && length(ex.args) == 2 && ex.args[2] isa Expr &&
        ex.args[2].head === :tuple
end

"The calls in `ast` to one of `names`, qualified or not, broadcast or not."
function calls_to(names, ast)
    collect_exprs(
        ex -> (ex.head === :call || is_broadcast(ex)) && callee(ex) in names, ast)
end

"Whether `ex` refers to the symbol `s` anywhere."
mentions(ex, s) = ex === s || (ex isa Expr && any(a -> mentions(a, s), ex.args))

"Whether `ex` is the keyword `name = <value>` with a value other than `false`."
function is_mark(ex, name)
    ex isa Expr && ex.head in (:kw, :(=)) && length(ex.args) == 2 &&
        ex.args[1] === name && ex.args[2] !== false
end

"Whether the macro call `m` is the device skip, `@test_skip Metal.functional()` or `@test_skip CUDA.functional()`."
function is_device_skip(m)
    callee(m) === Symbol("@test_skip") &&
        m.args[3:end] in ([:(Metal.functional())], [:(CUDA.functional())])
end

"""
The lines of a test file that mark a test as broken or skipped, as `line => what`: each
`@test_broken` and `@test_skip`, each `@test` with a `broken =` or `skip =` keyword, and each line
with a `broken = <value>` keyword of a call or a named tuple, such as Aqua's
`piracies = (; broken = true)`. A keyword line comes from the text, because an expression inside a
call carries no line of its own.

In a device file, `device = true`, the device skip `@test_skip Metal.functional()` or
`@test_skip CUDA.functional()` is no mark: it needs no issue.
"""
function broken_marks(ast, lines; device = false)
    marks = Dict{Int, String}()
    for m in collect_exprs(ex -> ex.head === :macrocall, ast)
        m.args[2] isa LineNumberNode || continue
        name, i = callee(m), m.args[2].line
        device && is_device_skip(m) && continue
        if name in (Symbol("@test_broken"), Symbol("@test_skip"))
            marks[i] = string(name)
        elseif name === Symbol("@test")
            for kw in (:broken, :skip)
                any(a -> is_mark(a, kw), m.args[3:end]) && (marks[i] = "$kw =")
            end
        end
    end
    keyword = collect_exprs(ast) do ex
        (ex.head === :kw && is_mark(ex, :broken)) ||
            (ex.head in (:tuple, :parameters) && any(a -> is_mark(a, :broken), ex.args))
    end
    if !isempty(keyword)
        for (i, l) in enumerate(lines)
            occursin(r"^[^#]*\bbroken\s*=(?!=)(?!\s*false\b)", l) && (marks[i] = "broken =")
        end
    end
    return marks
end

"The violations of the test convention in `repo`, one string per violation."
function violations(repo)
    out = String[]
    name = splitpath(normpath(repo))[end]
    v(rule, what) = push!(out, "$name: [$rule] $what")
    t = joinpath(repo, "test")
    rel(p) = relpath(p, t)
    rtfile = joinpath(t, "runtests.jl")

    # D1: the test dependencies are in test/Project.toml, and nowhere else
    projfile = joinpath(repo, "Project.toml")
    if isfile(projfile)
        proj = read(projfile, String)
        for sec in ("extras", "targets")
            has_section(proj, sec) &&
                v("D1", "Project.toml has a [$sec] section")
        end
    else
        v("D1", "Project.toml does not exist")
    end
    isfile(rtfile) || (v("D2", "test/runtests.jl does not exist"); return out)
    testproj = joinpath(t, "Project.toml")
    isfile(testproj) || v("D1", "test/Project.toml does not exist")

    # compat: test/ and docs/ carry no bound for a dependency that the package has too
    if isfile(projfile)
        root = TOML.parsefile(projfile)
        shared = union(keys(get(root, "deps", Dict())), keys(get(root, "weakdeps", Dict())))
        for sub in ("test", "docs")
            p = joinpath(repo, sub, "Project.toml")
            isfile(p) || continue
            c = get(TOML.parsefile(p), "compat", Dict())
            for d in sort!(collect(intersect(keys(c), shared)))
                v("compat",
                    "$sub/Project.toml has a [compat] entry for $d, a dependency of Project.toml")
            end
        end
    end

    L = layout(repo)
    for f in L.parse_errors
        v("D2", "$(rel(f)) does not parse")
    end
    isempty(L.parse_errors) || return out

    # D2, D5: runtests.jl is `using SafeTestsets`, GROUPS, and one `if` per group of @safetestset;
    # D5, ci-groups: the groups and the form of GROUPS, as `groups_violations` judges them
    for (rule, what) in groups_violations(repo, L.runtests)
        v(rule, what)
    end
    seen = String[]
    line = 0
    for s in L.runtests.args
        s isa LineNumberNode && (line = s.line; continue)
        if s == :(using SafeTestsets)
            "using" in seen &&
                v("D2", "runtests.jl:$line: `using SafeTestsets` is repeated")
            push!(seen, "using")
        elseif s isa Expr && s.head === :const && s.args[1] isa Expr &&
               s.args[1].head === :(=) && s.args[1].args[1] === :GROUPS
            "GROUPS" in seen && v("D5", "runtests.jl:$line: GROUPS is defined again")
            push!(seen, "GROUPS")
        elseif s isa Expr && s.head === :if && group_of(s.args[1]) !== nothing
            g = group_of(s.args[1])
            g in seen && v("D2", "group \"$g\" has more than one `if`")
            push!(seen, g)
            length(s.args) > 2 && v("D2", "the `if` of group \"$g\" has an `else`")
            bline = line
            for b in s.args[2].args
                b isa LineNumberNode && (bline = b.line; continue)
                ok = b isa Expr && b.head === :macrocall &&
                     b.args[1] === Symbol("@safetestset") &&
                     length(b.args) == 4 && b.args[4] isa Expr &&
                     b.args[4].head === :call && b.args[4].args[1] === :include
                ok || v("D2",
                    "runtests.jl:$(describe(b, bline)) in group \"$g\" is not `@safetestset \"<label>\" include(\"<path>\")`")
            end
        else
            v("D2", "runtests.jl:$(describe(s, line)) is outside the convention")
        end
    end
    "using" in seen || v("D2", "runtests.jl has no `using SafeTestsets`")
    "GROUPS" in seen || v("D5", "runtests.jl does not define GROUPS")
    for n in L.nested
        v("D2", "runtests.jl includes a nested runtests.jl: $(rel(n))")
    end
    for (file, line) in L.nonliteral
        v("D2", "$(rel(file)):$line: `include` of a path that is not a literal")
    end
    # an entry of the top-level runtests.jl is judged above, statement by statement
    for e in L.entries
        e.file == rtfile || e.kind === :safetestset ||
            v("D2",
                "$(rel(e.path)) is reached by a $(e.kind === :bare ? "bare `include`" : "`@testset` around an `include`"), not `@safetestset`")
    end

    # label: a plain string, unique in the file
    labels = [e.label for e in L.entries if e.kind === :safetestset]
    for l in labels
        l isa String || v("label", "the label $(l) is not a plain string")
    end
    for l in unique(filter(l -> l isa String && count(==(l), labels) > 1, labels))
        v("label", "the label \"$l\" is used more than once")
    end

    # D10: a file in `broken` names its issue on its line
    srclines = Dict{String, Vector{String}}()
    for e in L.entries
        e.group == "broken" &&
            !occursin(ISSUE, get!(() -> readlines(e.file), srclines, e.file)[e.line]) &&
            v("D10", "$(rel(e.path)) is in `broken` with no issue on its line")
    end

    # D8: every file under test/ is listed once, or is under test/helpers/
    listed = [e.path for e in L.entries]
    for p in unique(listed)
        count(==(p), listed) > 1 && v("D8", "$(rel(p)) is listed more than once")
        isfile(p) || v("D8", "$(rel(p)) is listed and does not exist")
        startswith(rel(p), "helpers/") && v("D8", "$(rel(p)) is a helper and is listed")
    end
    for f in L.files
        startswith(rel(f), "helpers/") || f in listed || f in L.nested ||
            v("D8", "$(rel(f)) is not listed in runtests.jl")
    end

    # D4, D9: the quality files
    q(f) = normpath(joinpath(t, "quality", f))
    isfile(q("aqua.jl")) || v("D4", "test/quality/aqua.jl does not exist")
    src = joinpath(repo, "src")
    docs = vcat(files_in(src, ".jl"), files_in(joinpath(repo, "docs", "src"), ".md"))
    if any(f -> occursin("jldoctest", read(f, String)), docs) && !isfile(q("doctests.jl"))
        v("D9", "the package has doctests and test/quality/doctests.jl does not exist")
    end
    for e in filter(e -> e.path == q("doctests.jl"), L.entries)
        e.group == "doctests" ||
            v("D9",
                "test/quality/doctests.jl is in group \"$(something(e.group, "none"))\", not \"doctests\"")
    end
    for e in filter(e -> e.group == "doctests" && e.path != q("doctests.jl"), L.entries)
        v("D9", "$(rel(e.path)) is in group \"doctests\", which holds only test/quality/doctests.jl")
    end

    # D3: test/<path>.jl sits where src/<path>.jl does; a top-level test/<name>.jl has a
    # src/<name>.jl of that exact name (`readdir`: `isfile` ignores case on macOS)
    for f in L.files
        isdir(src) || break
        r = rel(f)
        first(splitpath(r)) in OWN_DIRS && continue
        d = dirname(r)
        if isempty(d)
            r in readdir(src) && isfile(joinpath(src, r)) ||
                v("D3", "$r is at the top level of test/ and mirrors no src/$r")
        else
            isdir(joinpath(src, d)) || v("D3", "$r has no directory src/$d to mirror")
        end
    end

    # D5, D7, using, seed: what each test file does
    for f in L.files
        r = rel(f)
        ast = parsefile(f)
        ast === nothing && (v("D2", "$r does not parse"); continue)
        conds = collect_exprs(ex -> ex.head in (:if, :elseif, :&&, :||), ast)
        # a test file has no use for ARGS; ENV may set an option, but not decide what runs
        # the `skip =` and `broken =` keywords of a `@test` decide whether it runs, too
        kws = collect_exprs(ex -> ex.head === :(=) && ex.args[1] in (:skip, :broken), ast)
        (mentions(ast, :ARGS) || any(c -> mentions(c.args[1], :ENV), conds) ||
         any(k -> mentions(k.args[2], :ENV), kws)) &&
            v("D5", "$r reads ARGS or ENV to decide what runs")
        lines = readlines(f)
        device = startswith(r, "devices/")
        for (i, what) in sort!(collect(broken_marks(ast, lines; device)))
            occursin(ISSUE, lines[i]) ||
                v("D7", "$r:$i has `$what` with no issue on its line")
        end
        startswith(r, "helpers/") && continue
        isempty(collect_exprs(ex -> ex.head in (:using, :import), ast)) &&
            v("using", "$r has no `using` of its own")
        !isempty(calls_to(RANDOM_CALLS, ast)) && isempty(calls_to(SEED_CALLS, ast)) &&
            v("seed", "$r draws random numbers with no fixed seed")
    end
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2 || ARGS[1] != "--check"
        println(stderr, "usage: julia --startup-file=no test-layout.jl --check <repository> ...")
        exit(2)
    end
    # print each repository as it is checked
    found = sum(ARGS[2:end]) do r
        vs = violations(abspath(r))
        foreach(println, vs)
        length(vs)
    end
    exit(found == 0 ? 0 : 1)
end
