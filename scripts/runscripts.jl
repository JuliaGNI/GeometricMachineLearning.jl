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
# `SKIPPED` is the one exception, and it is closed in both directions: an entry naming a file that
# no longer exists, or that is no longer an entry point, fails this driver. An entry is a backlog
# item rather than a design — it says what stops the script running here, not that it is fine.
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

# A smoke run of a reproduction script takes seconds. The ceiling is here so that a script which
# hangs is named, rather than killing the whole job at the runner's timeout with nothing to read.
const TIMEOUT_SECONDS = 900

# The two entries left are the scripts whose subject *is* the GPU path: they name CUDA types
# directly rather than taking a backend, so there is nothing for a smoke run to switch. Every other
# GPU script takes its backend from `smoke_size`, runs on the CPU here, and is gated.
const SKIPPED = Dict{String, String}(
    "reproduction/linear_symplectic_transformer_gpu.jl" => "constructs all three networks on a `CUDABackend()` written into the call; no CI runner has a GPU",
    "reproduction/sympnets/sympnet_pendulum_cuda.jl" => "calls `CUDA.device()` and `CUDA.zeros` directly, which is what distinguishes it from `sympnet_pendulum.jl`; no CI runner has a GPU",
    "reproduction/symplectic_autoencoders/training.jl" => "its hand-rolled batch loop calls `dl.batch_size` and `redraw_batch!(dl)`, neither of which a `DataLoader` has: batching is `Batch` and the `Optimizer` functor now, so what is left is a rewrite of the loop rather than a repair",
    "reproduction/symplectic_autoencoders/online_sympnet.jl" => "training the reduced integrator raises `Functor not defined for NetworkLoss` -- `ReducedLoss` binds its input and output to one type parameter, so a pair that does not match exactly reaches the fallback; the repair is in src/loss/losses.jl",
    "reproduction/symplectic_autoencoders/online_transformer_for_sae.jl" => "the same `ReducedLoss` fallback, reached through the same `Optimizer` call"
)

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
function run_script(relative::AbstractString; smoke::Bool)
    path = joinpath(ROOT, relative)
    environment = copy(ENV)
    smoke && (environment["GML_SMOKE"] = "1")
    command = Cmd(`$(Base.julia_cmd()) --startup-file=no --project=$PROJECT $path`;
        dir = dirname(path), env = environment)

    log = tempname()
    started = time()
    process = open(log, "w") do io
        p = run(pipeline(command; stdout = io, stderr = io); wait = false)
        timer = Timer(_ -> (process_running(p) && kill(p, Base.SIGKILL)), TIMEOUT_SECONDS)
        wait(p)
        close(timer)
        return p
    end
    elapsed = time() - started

    # A process killed by the timer exits with status 0 and a non-zero `termsignal`, so the exit
    # code alone reads a timeout as success.
    verdict = if process.termsignal != 0
        "TIMEOUT"
    elseif process.exitcode != 0
        "FAILED"
    else
        "ok"
    end
    return verdict, elapsed, read(log, String)
end

function main(mode::AbstractString)
    mode in ("verification", "reproduction") ||
        error("usage: runscripts.jl verification|reproduction")
    smoke = mode == "reproduction"

    # The skip list may not outlive what it excuses.
    runnable = vcat(entry_points("verification"), entry_points("reproduction"))
    stale = filter(f -> !(f in runnable), sort!(collect(keys(SKIPPED))))
    if !isempty(stale)
        println("skip list is stale -- these are no longer entry points under verification/ or " *
                "reproduction/:")
        foreach(f -> println("  ", f), stale)
        return 1
    end

    failures = String[]
    for relative in entry_points(mode)
        if haskey(SKIPPED, relative)
            println("skip     ", relative, "  -- ", SKIPPED[relative])
            flush(stdout)
            continue
        end
        verdict, elapsed, output = run_script(relative; smoke = smoke)
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
