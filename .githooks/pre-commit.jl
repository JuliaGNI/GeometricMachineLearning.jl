# The stages of the shared pre-commit hook. Canonical copy: githooks/pre-commit.jl of the
# harness. Installed beside its launcher, `pre-commit`, into each repository's .githooks/ by
# harness githooks — edit the canonical copy and re-run the installer, never the copy in a
# repository.
#
# One Julia process runs every stage except the load test, which runs in a child process:
#
#   0. test layout              BLOCKS   on a commit that touches test/, any file type
#   1. JuliaFormatter           BLOCKS   formatting is mechanical and always fixable
#   2. fatou lint               WARNS    see `lint`
#   2.5 unicode NFC             BLOCKS   the tree is NFC, so a non-NFC file is a regression
#   3. package load             BLOCKS   catches a syntax error or a broken include
#
# Stages 1–3 run on STAGED .jl files only. A whole-tree JuliaFormatter run is not sure to end:
# one has died inside SciML's `nest!` after 62 minutes of CPU.
#
# Included rather than run as the program file, this file defines its functions and runs nothing.

using Unicode

const RED, YELLOW, GREEN, DIM, NC = "\e[0;31m", "\e[0;33m", "\e[0;32m", "\e[2m", "\e[0m"

"The `lines` joined, each indented by four spaces."
indent(lines) = join(("    " * l for l in lines), "\n")

"""
The paths that `git diff --cached --name-only` lists with `args`, or `nothing` when git fails.
An empty list from a git that failed would read as "nothing staged" and pass every commit, so
the caller refuses on `nothing`.
"""
function staged(args...)
    out = IOBuffer()
    p = run(pipeline(ignorestatus(`git diff --cached --name-only -z $args`); stdout = out))
    success(p) || return nothing
    return filter(!isempty, split(String(take!(out)), '\0'))
end

"""
Run `cmd` with its stdout and stderr in one stream. After `seconds`, send it SIGTERM, and
SIGKILL 5 s later if it still runs. Returns the exit code, or `nothing` when the time ran out,
and the output.
"""
function run_boxed(cmd, seconds)
    out = Pipe()
    p = run(pipeline(ignorestatus(cmd); stdin = devnull, stdout = out, stderr = out); wait = false)
    close(out.in)
    reader = @async read(out, String)
    timedout = Ref(false)
    stop = Timer(seconds) do _
        timedout[] = true
        kill(p)
        Timer(5) do _
            process_running(p) && kill(p, Base.SIGKILL)
        end
    end
    wait(p)
    close(stop)
    text = fetch(reader)
    return timedout[] ? nothing : p.exitcode, text
end

"""
Stage 0. A commit that touches test/ — any file, a deletion too — must leave the tests in the
layout of the test convention. `test-layout.jl` is installed beside this program by harness
githooks; a missing copy blocks rather than skips, because a stage that skips itself reports
"clean" when it checked nothing. It checks the working tree, not the index.

The script goes into a module of its own and its `violations` runs here: its command-line block
calls `exit`, and runs only when the script is the program file.
"""
function test_layout(root)
    script = joinpath(root, ".githooks", "test-layout.jl")
    if !isfile(script)
        println("$(RED)test layout: .githooks/test-layout.jl is missing$(NC) — re-run harness githooks --apply.")
        return false
    end
    found = try
        layout = Module(:TestLayout)
        Base.include(layout, script)
        @invokelatest layout.violations(root)
    catch e
        [sprint(showerror, e)]
    end
    if isempty(found)
        println("$(GREEN)test layout: ok$(NC)")
        return true
    end
    println("$(RED)test layout: FAILED$(NC) — the tests are not in the layout of the convention:")
    println(indent(split(join(found, "\n"), '\n')))
    return false
end

"""
Stage 1. Honours this repository's own .JuliaFormatter.toml. A hook cannot reach an MCP
server, so this calls JuliaFormatter directly, from the harness Julia environment.

Checked one file at a time: `format(files)` returns a single Bool for the whole batch, so its
message could only name every staged file. Per file costs nothing: the batch call formats each
file on its own anyway, and `.JuliaFormatter.toml` is found by searching upward from the file
either way.
"""
function formatting(files)
    bad = try
        env = get(ENV, "RESEARCH_HARNESS_JULIA",
            joinpath(homedir(), ".local", "share", "research-harness", "julia"))
        pushfirst!(LOAD_PATH, env)
        id = Base.identify_package("JuliaFormatter")
        id === nothing &&
            error("JuliaFormatter is in no environment of the load path; $env is first")
        formatter = Base.require(id)
        version = pkgversion(formatter)
        version !== nothing && version >= v"2.14" ||
            error("JuliaFormatter $version is older than 2.14, which the hook needs; $env pins it")
        format = formatter.format
        filter(f -> !Base.invokelatest(format, f; overwrite = false, throw_on_error = true), files)
    catch e
        println(stderr, sprint(showerror, e))
        files
    end
    if isempty(bad)
        println("$(GREEN)formatting: ok$(NC)")
        return true
    end
    println(stderr, "not formatted:\n", join(bad, "\n"))
    println("$(RED)formatting: FAILED$(NC) — run JuliaFormatter over the files above, then re-stage.")
    return false
end

"""
Stage 2, advisory. `fatou lint` never blocks, because it produces findings that are wrong.
`unused-import` does not follow `include`, so it flags load-bearing imports in every module
file; `docstring-argument-mismatch` cannot see that `f(::Type{T}, s)` binds `T`. Both can be
silenced one at a time: `# fatou-ignore <rule>` on its own line, directly above the finding,
silences it. A trailing comment on the same line does nothing, and `fatou.toml` rejects
`per-file-ignores`.

`parse-error` is what keeps this stage advisory. It is severity `error`, it cannot be
suppressed, and all 37 findings measured across 32 packages are false. Fatou rejects two
constructs that Julia parses: a parenthesised macro call whose body is a `for` block, as in
`(@allocated for ... end) ÷ n`, and a `const` whose assignment is on the next line.

fatou writes its findings and its summary to stderr, not stdout, and exits 1 when it has any,
so the verdict comes from the exit status and both streams are read. With findings there is no
summary line, so every line is one finding.

Time-boxed: `fatou lint` does not always end. On a generated source file of 25 MB and 91 625
lines, two runs each burned a core for six and a half hours. An advisory stage that hangs is
worse than one that fails, because a hung commit looks like nothing happening rather than like
a refusal. 60 s is about 600 times the measured whole-tree time of a large package.
"""
function lint(files)
    if Sys.which("fatou") === nothing
        println("$(YELLOW)fatou lint: skipped — fatou is not on the PATH$(NC)")
        return
    end
    # A fatou on the PATH that the system cannot start, such as a script whose interpreter is
    # missing, makes the start throw; the stage is advisory, so it is skipped.
    code, out = try
        run_boxed(`fatou lint --force-exclude --output concise $files`, 60)
    catch e
        e isa Base.IOError || rethrow()
        println("$(YELLOW)fatou lint: could not start — stage skipped, not a failure$(NC)")
        println(indent(split(sprint(showerror, e), '\n')))
        return
    end
    if code === nothing
        println("$(YELLOW)fatou lint: timed out after 60s — stage skipped, not a failure$(NC)")
        println("    A staged file is pathological for fatou.")
    elseif code == 0
        println("$(GREEN)fatou lint: clean$(NC)")
    else
        findings = filter(!isempty, split(out, '\n'))
        println("$(YELLOW)fatou lint: $(length(findings)) finding(s) — advisory, not blocking$(NC)")
        isempty(findings) || println(indent(findings))
    end
end

"""
Stage 2.5. Sources here are NFC-normalised: `ū`, `ḡ`, `ṗ`, `ẋ` as a single codepoint rather
than a base letter plus a combining mark. Julia's parser normalises identifiers to NFC, so an
NFD file compiles to byte-identical symbols and nothing about dispatch changes — but every
byte-matching tool (grep, ripgrep, an editor search, an agent's string replacement,
Documenter's doctest comparison) then fails to match a pattern typed in NFC, silently.

Blocking, because the invariant holds tree-wide: every tracked file in Packages/ and
Experiments/ is NFC, so anything this stage reports is a regression rather than a backlog.

The invariant is `s == Unicode.normalize(s, :NFC)`, never the absence of combining marks:
`q̇`, `v̄` and `f̄` have no precomposed codepoint and stay two codepoints under NFC.

The check is here rather than a call of ~/Research/Harness/githooks/nfc.jl, as the other stages
are: a hook that skips itself because a path outside the repository is missing reports "clean"
when it checked nothing. `nfc.jl --apply` fixes what this reports.
"""
function nfc(files)
    bad = filter(files) do f
        s = read(f, String)
        isvalid(s) && s != Unicode.normalize(s, :NFC)
    end
    if isempty(bad)
        println("$(GREEN)unicode: NFC$(NC)")
        return true
    end
    println("$(RED)unicode: $(length(bad)) file(s) not NFC$(NC)")
    println(indent(bad))
    println("    $(DIM)fix: julia --startup-file=no ~/Research/Harness/githooks/nfc.jl --apply <files>$(NC)")
    return false
end

"The `name` of `Project.toml` in the current directory, or `nothing` for none."
function package_name()
    isfile("Project.toml") || return nothing
    for line in eachline("Project.toml")
        m = match(r"^name\s*=\s*\"(.*)\"", line)
        m === nothing || return m[1]
    end
    return nothing
end

"""
Stage 3, only for an actual package: an experiment repository has a Project.toml without a name.

The package loads in a child process started with `Base.julia_cmd()`, the binary of this
process, in the repository's own environment: JuliaFormatter and most packages share
PrecompileTools and Preferences, and in one process the one loaded second gets the first one's
versions.

The output is captured from the one run and filtered rather than tailed: for the failure this
stage exists to catch, the cause — `ERROR: LoadError` and the file:line — is at the top, and a
tail keeps backtrace frames and the "Failed to precompile" summary while it drops the cause.
Dropping the ` [n] frame` and `  @ Module` lines leaves the message and its location in the
first lines.
"""
function load(pkg)
    out = IOBuffer()
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=. -e "using $pkg"`
    p = run(pipeline(ignorestatus(cmd); stdin = devnull, stdout = out, stderr = out))
    if success(p)
        println("$(GREEN)loads: ok$(NC)")
        return true
    end
    println("$(RED)loads: FAILED$(NC) — `using $pkg` errors:")
    lines = filter(l -> !occursin(r"^\s*\[\d+\]\s|^\s*@ ", l), split(String(take!(out)), '\n'))
    println(indent(first(lines, 20)))
    return false
end

function refuse_unlisted()
    println("\n$(RED)pre-commit: could not list staged files.$(NC) Refusing to pass silently.\n")
    exit(1)
end

function main()
    root = try
        readchomp(`git rev-parse --show-toplevel`)
    catch
        exit(1)
    end
    cd(root)

    jl = staged("--diff-filter=ACMRT", "--", "*.jl")
    jl === nothing && refuse_unlisted()
    tests = staged("--", "test/")
    tests === nothing && refuse_unlisted()

    ok = isempty(tests) || test_layout(root)
    if isempty(jl)
        ok ||
            println("\n$(RED)pre-commit failed.$(NC) Use --no-verify only if you know why.\n")
        exit(ok ? 0 : 1)
    end

    println("\n$(DIM)pre-commit$(NC)  $(length(jl)) staged .jl file(s)")
    ok &= formatting(jl)
    lint(jl)
    ok &= nfc(jl)
    pkg = package_name()
    pkg === nothing || (ok &= load(pkg))

    if !ok
        println("\n$(RED)pre-commit failed.$(NC) Use --no-verify only if you know why.\n")
        exit(1)
    end
    println()
    exit(0)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
