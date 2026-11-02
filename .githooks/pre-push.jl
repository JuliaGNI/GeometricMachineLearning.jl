# The gate and the suite of the shared pre-push hook. Canonical copy: githooks/pre-push.jl of
# the harness. Installed beside its launcher, `pre-push`, into each repository's .githooks/ by
# harness githooks — edit the canonical copy and re-run the installer, never the copy in a
# repository.
#
# Runs the full test suite, but only when pushing to main/master. A suite on every push to every
# branch makes a push take 10-30 minutes with no output, which looks exactly like a network
# hang. A topic branch does not need the gate; CI has it.

using TOML

const RED, GREEN, DIM, NC = "\e[0;31m", "\e[0;32m", "\e[2m", "\e[0m"

"""
Whether a ref line from git — `<local ref> <local sha> <remote ref> <remote sha>`, one per ref
— pushes to `refs/heads/main` or `refs/heads/master`.
"""
function gated(line)
    fields = split(line)
    return length(fields) >= 3 && fields[3] in ("refs/heads/main", "refs/heads/master")
end

"""
The `name` of `Project.toml` in the current directory, of any type, when the file has both a
`name` and a `uuid`; `nothing` when it has no file, no `name` or no `uuid`, which is no package.
Throws `TOML.ParserError` for a `Project.toml` that does not parse.
"""
function package_name()
    isfile("Project.toml") || return nothing
    project = TOML.parsefile("Project.toml")
    haskey(project, "name") && haskey(project, "uuid") || return nothing
    return project["name"]
end

"""
Whether the package name `name` is a Julia identifier that the parser reads as exactly that name.
The suite's command puts the name into Julia source, so any other name could run code of its own.
`Base.isidentifier` accepts a keyword such as `end`; the parser of the running Julia refuses it.
"""
function identifier(name)
    name isa AbstractString && Base.isidentifier(name) &&
        Meta.parse(name; raise = false) === Symbol(name)
end

"""
The first two lines of the error `e`, joined with ` — `: its message and, for a parse error of
TOML, the line that gives the location.
"""
headline(e) = join(first(split(sprint(showerror, e), '\n'), 2), " — ")

"""
The suite's command, or `nothing` when there is nothing to run.

--check-bounds=auto matters. Up to Julia 1.12, `Pkg.test` runs the tests with
`--check-bounds=yes` whatever the calling process uses, which forces recompilation of every
dependency into a separate cache and diverges from what CI and a normal session use. That
divergence is the reason a test suite must not run concurrently with another heavy Julia job
over the same packages. So the flag goes to the test process through `julia_args`, as well as
to the process that calls `Pkg.test`.
"""
function suite(pkg)
    julia = `$(Base.julia_cmd()) --startup-file=no --check-bounds=auto --project=.`
    pkg === nothing ||
        return `$julia -e "using Pkg; Pkg.test(\"$pkg\"; julia_args = \`--check-bounds=auto\`)"`
    isfile(joinpath("test", "runtests.jl")) && return `$julia test/runtests.jl`  # not a package
    return nothing
end

"""
Run `cmd` in a process group of its own and wait for it; return whether it passed.

An interrupt of this program goes on to the whole group, the suite and the test process that
`Pkg.test` starts, and the program waits until no process of the group is left before it
exits: a test process that outlives its hook survives as an orphan, and such processes
accumulate over days. A group that still has a process `grace` seconds after the interrupt
gets SIGKILL, also when the suite itself has ended. A further interrupt during that wait does
not end it.

The suite starts, and its PID is read, with SIGINT deferred, inside the `try`: an interrupt
that comes during the start is thrown when the deferral ends, and finds `p` and `group` set.
"""
function run_suite(cmd; grace = 30)
    flush(stdout)
    p = nothing
    group = Cint(0)
    signal(s) = ccall(:kill, Cint, (Cint, Cint), -group, s)
    try
        disable_sigint() do
            p = run(
                pipeline(ignorestatus(Cmd(cmd; detach = true)); stdin = devnull, stdout, stderr);
                wait = false)
            # `detach` starts a new session, whose group ID is the PID of `p`. Read before
            # anything yields: once a program that ends at once is reaped, `getpid` throws.
            group = getpid(p)
        end
        wait(p)
    catch e
        e isa InterruptException || rethrow()
        # `p` is `nothing` for an interrupt that comes before the start.
        if p !== nothing
            signal(Base.SIGINT)
            killer = Timer(_ -> signal(Base.SIGKILL), grace)
            # `kill(-group, 0)` fails once the group is gone, and on macOS also for a group
            # whose only processes are zombies that nobody has reaped yet.
            while process_running(p) || signal(0) == 0
                try
                    process_running(p) ? wait(p) : sleep(0.1)
                catch e
                    e isa InterruptException || rethrow()
                end
            end
            close(killer)
        end
        println("\n$(RED)pre-push: interrupted; the suite has stopped.$(NC)\n")
        exit(130)
    end
    return success(p)
end

function main()
    Base.exit_on_sigint(false)
    root = try
        readchomp(`git rev-parse --show-toplevel`)
    catch
        exit(1)
    end
    cd(root)

    # Every line is read, so git never writes to a closed pipe.
    if !any(gated, readlines(stdin))
        println("\n$(DIM)pre-push$(NC)  not main/master — skipping the suite, CI will run it.\n")
        exit(0)
    end

    pkg = try
        package_name()
    catch e
        e isa TOML.ParserError || rethrow()
        println("\n$(RED)pre-push: Project.toml does not parse$(NC) — ", headline(e), "\n")
        exit(1)
    end
    if pkg !== nothing && !identifier(pkg)
        println("\n$(RED)pre-push: the name $(repr(pkg)) in Project.toml is not a Julia identifier$(NC)\n")
        exit(1)
    end
    println("\n$(DIM)pre-push$(NC)  pushing to main/master — running the full suite for ",
        something(pkg, basename(root)), ".")
    println("$(DIM)This takes 10-30 minutes and prints nothing until it finishes. That is not a hang.$(NC)\n")

    cmd = suite(pkg)
    if cmd === nothing
        println("$(DIM)no package name and no test/runtests.jl — nothing to run.$(NC)\n")
        exit(0)
    end
    if !run_suite(cmd)
        println("\n$(RED)ERROR — tests must pass before pushing to main or master.$(NC)\n")
        exit(1)
    end
    println("\n$(GREEN)Suite passed.$(NC)\n")
    exit(0)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
