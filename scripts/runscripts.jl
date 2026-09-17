# The gate for `scripts/`.
#
# `scripts/` is three directories with stated purposes:
#
#   verification/  checks that establish a mathematical claim about the package. Run in full.
#   reproduction/  the runs that produced the committed network weights and the manual's figures.
#                  Run in smoke mode, at the small sizes `utilities/smoke.jl` returns.
#   utilities/     what the other two include, plus the standalone tools. Never run on its own.
#
# Every `.jl` file under `verification/` and `reproduction/` is an entry point and is run here,
# each in its own process with its own directory as the working directory. There is no list of
# what to run, so a script added to either directory is gated from the moment it lands. A file
# that is included rather than run belongs in `utilities/`.
#
# `SKIPPED` is the mechanism for excluding one, and it is empty. It is closed in both directions:
# an entry naming a file that no longer exists, or that is no longer an entry point, fails this
# driver. An entry is a backlog item rather than a design — it says what stops the script running
# here, not that it is fine.
#
# Usage, from the repository root:
#
#   julia --startup-file=no --project=scripts scripts/runscripts.jl verification
#   julia --startup-file=no --project=scripts scripts/runscripts.jl reproduction
#
# The exit status is the verdict: 0 when every script ran to completion, 1 otherwise.

const ROOT = @__DIR__

# Each script runs in the environment this driver was started in, rather than in one named here:
# CI starts it with `--project=scripts` against a `scripts/Manifest.toml` it has just instantiated,
# and a local run can point at a scratch copy of that environment instead of resolving one inside
# the repository.
const PROJECT = Base.active_project()

# The ceiling is here so that a script which hangs is named, rather than killing the whole job at
# the runner's timeout with nothing to read. It is per mode because the two modes cost differently:
# a verification script runs in full, and a reproduction script runs at its smoke size and takes
# seconds, so five minutes there is already far past "still working".
const TIMEOUT_SECONDS = Dict("verification" => 900, "reproduction" => 300)

# The budget for a whole mode, inside `Scripts.yml`'s `timeout-minutes: 90` less what that job
# spends instantiating. Without it the per-script ceiling does not keep the promise above: 23
# reproduction scripts each entitled to the ceiling outlast any runner, and the job then dies at the
# runner's timeout with no verdict at all. The driver stops first, and names what it did not reach.
const BUDGET_SECONDS = 3600

# Empty, and meant to stay that way. An entry here is a script the gate does not cover, so the
# only honest reason for one is hardware this runner does not have -- never a defect left unfixed.
# Every GPU script takes its backend from `smoke_size`, runs on the CPU here, and is gated.
const SKIPPED = Dict{String, String}()

"""
Return every `.jl` file under `dir`, relative to `scripts/` and with `/` separators.
"""
function entry_points(dir::AbstractString)
    files = String[]
    for (root, _, names) in walkdir(joinpath(ROOT, dir)), name in names

        endswith(name, ".jl") &&
            push!(files, replace(relpath(joinpath(root, name), ROOT), '\\' => '/'))
    end
    return sort!(files)
end

"""
Run one script in its own process, with its own directory as the working directory. Return the
verdict, the elapsed seconds and the captured output.
"""
function run_script(relative::AbstractString; smoke::Bool, timeout::Real)
    path = joinpath(ROOT, relative)
    environment = copy(ENV)
    smoke && (environment["GML_SMOKE"] = "1")
    command = Cmd(`$(Base.julia_cmd()) --startup-file=no --project=$PROJECT $path`;
        dir = dirname(path), env = environment)

    log = tempname()
    started = time()
    process = open(log, "w") do io
        p = run(pipeline(command; stdout = io, stderr = io); wait = false)
        timer = Timer(_ -> (process_running(p) && kill(p, Base.SIGKILL)), timeout)
        wait(p)
        close(timer)
        return p
    end
    elapsed = time() - started
    output = read(log, String)
    rm(log; force = true)

    # A process killed by the timer exits with status 0 and a non-zero `termsignal`, so the exit
    # code alone reads a timeout as success.
    verdict = if process.termsignal != 0
        "TIMEOUT"
    elseif process.exitcode != 0
        "FAILED"
    else
        "ok"
    end
    return verdict, elapsed, output
end

function main(mode::AbstractString)
    mode in ("verification", "reproduction") ||
        error("usage: runscripts.jl verification|reproduction")
    smoke = mode == "reproduction"
    timeout = TIMEOUT_SECONDS[mode]

    # The skip list may not outlive what it excuses.
    runnable = vcat(entry_points("verification"), entry_points("reproduction"))
    stale = filter(f -> !(f in runnable), sort!(collect(keys(SKIPPED))))
    if !isempty(stale)
        println("skip list is stale -- these are no longer entry points under verification/ or " *
                "reproduction/:")
        foreach(f -> println("  ", f), stale)
        return 1
    end

    # Discovery finding nothing is a renamed or emptied directory, not a clean run. Without this the
    # loop below has nothing to fail on and the driver reports success.
    scripts = entry_points(mode)
    if isempty(scripts)
        println("no entry points under ", mode, "/ -- the directory is empty or has been renamed")
        return 1
    end

    deadline = time() + BUDGET_SECONDS
    failures = String[]
    for (index, relative) in enumerate(scripts)
        if time() ≥ deadline
            unreached = scripts[index:end]
            println("\nthe ", BUDGET_SECONDS, "s budget is spent; ", length(unreached),
                " script(s) were not run:")
            foreach(f -> println("  ", f), unreached)
            append!(failures, unreached)
            break
        end
        if haskey(SKIPPED, relative)
            println("skip     ", relative, "  -- ", SKIPPED[relative])
            flush(stdout)
            continue
        end
        verdict, elapsed, output = run_script(relative; smoke = smoke, timeout = timeout)
        println(rpad(verdict, 9), rpad(string(round(elapsed; digits = 1)) * "s", 9), relative)
        if verdict != "ok"
            push!(failures, relative)
            println(output)
        end
        flush(stdout)
    end

    if isempty(failures)
        println("\nall ", mode, " scripts ran to completion")
        return 0
    end
    println("\n", length(failures), " script(s) did not run to completion:")
    foreach(f -> println("  ", f), failures)
    return 1
end

exit(main(isempty(ARGS) ? "" : ARGS[1]))
