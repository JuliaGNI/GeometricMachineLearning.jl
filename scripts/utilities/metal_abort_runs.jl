# The counted runs of GeometricMachineLearning's two Metal process aborts.
#
# Two Julia processes died on Metal in `Float32`, each in the first Adam call of a structured
# network: `LASympNet(4)` with a `SIGABRT` from `MTLReleaseAssertionFailure` in
# `-[IOGPUMetalCommandBuffer setCurrentCommandEncoder:]` while a second `ex` read a Julia vector,
# and `VolumePreservingFeedForward(4)` with a `SIGTRAP` from `BUG IN CLIENT OF LIBMALLOC: memory
# corruption of free block` inside the host LLVM JIT, with no second `ex`. `KNOWN_ISSUES.md` K35
# records the counts.
#
# This script is the procedure that asks whether either recurs. Every mode is one program, so that
# the number of runs, the fields of a run's log and the rule that turns logs into a count are fixed
# in one place and cannot drift between the arms.
#
# ## The arms
#
# * `lasympnet` — `LASympNet(4)`, no concurrent evaluation.
# * `vpff` — `VolumePreservingFeedForward(4)`, no concurrent evaluation.
# * `lasympnet-concurrent` — `LASympNet(4)` with a second `ex` reading a host `Vector{Float32}`
#   while the training call runs. Only a Kaimon session can run it, because the concurrency is two
#   requests to one session: `metal_abort_session.jl` holds its training half and its reading half.
#   `run` cannot start it.
#
# ## A run
#
# One OS process, never reused. It sets `MTL_DEBUG_LAYER=1` before `using Metal`, prints its pid,
# `Threads.nthreads()`, the Julia version, the version and path of every package of the stack, then
# `Metal.functional()` and `Metal.device()` with its name. An absent or non-functional device ends
# the run before any Metal work. It builds a network on `MetalBackend()` in `Float32`, takes a
# forward pass and a Zygote pullback on it, builds a *new* network and takes the first training call
# on that one — Adam, 2 epochs, `Batch(8)`, `MtlArray(randn(Float32, 4, 40))` — timed as
# `@timed Base.invokelatest(call)`, so that the compilation is inside the timer. A run whose
# `compile_time` is 0 is a warm call, and void.
#
# ## What counts
#
# An abort is a death by a signal together with a new `julia-*.ips` file in
# `~/Library/Logs/DiagnosticReports` whose `"pid"` is the run's. The reports directory holds reports
# from other Julia processes, so a count of new files alone miscounts: every run records the report
# files that existed before it and after it, and only a report whose pid is the run's own is
# attached to it. Each attached report is copied beside the run's log, because macOS deletes these
# reports. In the `lasympnet-concurrent` arm such a death is an abort only when the log also holds
# the `M8-TRAIN-START` and `M8-READ-START` records of an overlap, see `overlap_evidence`; without
# them it is void.
#
# A report reproduces the `LASympNet` abort when it carries `setCurrentCommandEncoder`,
# `commitEncoder`, `endEncoding` or `endComputePass`, and the `VolumePreservingFeedForward` abort
# when its text carries `BUG IN CLIENT OF LIBMALLOC`. The heap was corrupt before the frame that
# detected it ran, so that frame locates the detection only.
#
# ## N
#
# `RUNS_PER_ARM` is 59, fixed here before the first counted run. At two aborts in about 40 training
# calls the rate per call is 1/20, and `0.95^59 = 0.0485 ≤ 0.05`: 59 clean runs make a rate of 1/20
# or more unlikely at the 5 per cent level. An arm that stops below 59 proves nothing.
#
# ## Usage
#
# The environment is a scratch one that develops this package and the local GeometricOptimizers, and
# has Metal. `scripts/Project.toml` does not carry Metal, and CI runs only `verification/` and
# `reproduction/`, so nothing here is run by a job.
#
#     julia --startup-file=no --project=<env> scripts/utilities/metal_abort_runs.jl calibrate --out <dir>
#     julia --startup-file=no --project=<env> scripts/utilities/metal_abort_runs.jl run --arm lasympnet --runs 1:59 --out <dir>
#     julia --startup-file=no --project=<env> scripts/utilities/metal_abort_runs.jl audit --out <dir>
#     julia --startup-file=no scripts/utilities/metal_abort_runs.jl selftest
#
# `--threads <n>` sets the child's `Threads.nthreads()` and defaults to `THREADS`, which is what the
# Kaimon session prints. `--reports <dir>` moves the reports directory, which `selftest` uses and a
# counted run does not. `audit` reads the logs and prints the count of every arm; it is the only
# thing that decides an arm's verdict, so a critic recounts from the same logs.

const RUNS_PER_ARM = 59
const THREADS = 16
const REPORTS = joinpath(homedir(), "Library", "Logs", "DiagnosticReports")
const ARMS = ("lasympnet", "vpff", "lasympnet-concurrent")
const ENCODER_FRAMES = (
    "setCurrentCommandEncoder", "commitEncoder", "endEncoding", "endComputePass")
const CRASH_REPORTER_WAIT = 60

# ---------------------------------------------------------------- the report directory

"The report files that the crash reporter holds, by name."
function report_files(dir::AbstractString = REPORTS)
    isdir(dir) || return String[]
    sort!(filter(f -> startswith(f, "julia-") && endswith(f, ".ips"), readdir(dir)))
end

"The process id in a crash report, from the one `\"pid\"` field of its header."
function report_pid(path::AbstractString)
    m = match(r"\"pid\"\s*:\s*(\d+)", read(path, String))
    m === nothing ? nothing : parse(Int, m.captures[1])
end

"""
Which of the two aborts a crash report shows, from its text: `:malloc` for the host-heap corruption
of the `VolumePreservingFeedForward` run, `:encoder` for a command-buffer or command-encoder frame
of the `LASympNet` run in a report that names Metal anywhere (in practice the `Metal.framework`
image path, since no frame symbol carries the word), and `:other` for a fault of neither kind.

Both are read from the whole report, because a fault on any thread is a fault of the process and the
report's faulting thread is not always the one that matters.
"""
function report_signature(path::AbstractString)
    text = read(path, String)
    occursin("BUG IN CLIENT OF LIBMALLOC", text) && return :malloc
    any(f -> occursin(f, text), ENCODER_FRAMES) && occursin("Metal", text) &&
        return :encoder
    :other
end

"The first `n` symbol names of a crash report, for the record beside the run."
function report_symbols(path::AbstractString; n::Int = 12)
    symbols = [m.captures[1]
               for m in eachmatch(r"\"symbol\"\s*:\s*\"([^\"]*)\"", read(path, String))]
    filter!(!isempty, symbols)
    symbols[1:min(n, length(symbols))]
end

# ---------------------------------------------------------------- the run log

"One `M8-` line, flushed, so that a run that dies leaves every line it printed."
function emit(io::IO, tag::AbstractString, fields::Pair...)
    print(io, "M8-", tag)
    for (k, v) in fields
        print(io, " ", k, "=", v)
    end
    println(io)
    flush(io)
end

_no_space(x) = replace(string(x), ' ' => '_')

"The value of `key=` in one log line, or `nothing`."
function value_of(line::AbstractString, key::AbstractString)
    m = match(Regex("(?<![\\w-])" * key * "=([^\\s]*)"), line)
    m === nothing ? nothing : m.captures[1]
end

"""
The fields of one run's log, as this script reads them back. `concurrent` is whether an `M8-READ`
line of the session arm says that its reads overlapped the training call, and `overlap` is whether
the records written before the timed call finished show it, see `overlap_evidence`. `reports` holds
the crash reports that carry this run's pid, either as the parent attached them or as `audit` found
them later.
"""
struct RunLog
    arm::String
    index::Int
    pid::Int
    exit::Int
    endline::Union{Nothing, String}
    functional::Bool
    layer::String
    validation::Bool
    concurrent::Bool
    overlap::Bool
    compile_time::Float64
    finite::Bool
    reports::Vector{NamedTuple{
        (:file, :pid, :match, :signature), Tuple{String, Int, Bool, Symbol}}}
end

"The loaded module of a name, whether the active project has it as a dependency or a package loaded it."
function loaded_module(name::AbstractString)
    for (id, m) in Base.loaded_modules
        id.name == name && return m
    end
    nothing
end

"The version and path of every package of the stack, one `M8-PKG` line each."
function print_stack(io::IO)
    for name in ("GeometricMachineLearning", "GeometricOptimizers", "Metal", "GPUCompiler",
        "KernelAbstractions", "LLVM")
        m = loaded_module(name)
        if m === nothing
            emit(io, "PKG", :name => name, :version => "-", :path => "-")
            continue
        end
        emit(io, "PKG", :name => name, :version => string(pkgversion(m)),
            :path => _no_space(pkgdir(m)))
    end
end

# ---------------------------------------------------------------- one run

"""
One counted run, in the process that started for it: the device and its name, the stack, the forward
pass and the pullback on one network, and the first training call of a new one, timed.
"""
function single(arm::AbstractString)
    t0 = time()
    emit(stdout, "RUN", :arm => arm, :pid => getpid(),
        :time => t0, :concurrent => false)
    layer = get(ENV, "MTL_DEBUG_LAYER", "-")
    emit(stdout, "ENV", :MTL_DEBUG_LAYER => "\"$layer\"", :julia => VERSION,
        :nthreads => Threads.nthreads(), :project => _no_space(Base.active_project()))

    print_stack(stdout)

    if layer != "1"
        emit(stdout, "END", :status => "error", :reason => "MTL_DEBUG_LAYER_is_not_1")
        exit(2)
    end
    if !Metal.functional()
        emit(stdout, "DEVICE", :functional => false)
        emit(stdout, "END", :status => "error", :reason => "no_functional_Metal_device")
        exit(3)
    end
    device = Metal.device()
    name = try
        _no_space(device.name)
    catch
        "-"
    end
    emit(stdout, "DEVICE", :functional => true, :type => _no_space(nameof(typeof(device))),
        :name => name)

    Random.seed!(2024)
    X = MtlArray(randn(Float32, 4, 40))
    build = startswith(arm, "vpff") ? () -> VolumePreservingFeedForward(4) :
            () -> LASympNet(4)

    nn = NeuralNetwork(build(), MetalBackend(), Float32)
    y = nn(X)
    emit(stdout, "FORWARD", :eltype => eltype(y), :size => _no_space(size(y)))
    loss(ps) = sum(abs2, nn.model(X, ps))
    value, back = Zygote.pullback(loss, GeometricMachineLearning.params(nn))
    pullback = back(one(value))
    emit(stdout, "PULLBACK", :value => value, :leaves => length(pullback))

    trained = NeuralNetwork(build(), MetalBackend(), Float32)
    optimizer = TrainingOptimizer(GeometricMachineLearning.params(trained); algorithm = Adam())
    loader = DataLoader(X; autoencoder = false, suppress_info = true)
    batch = Batch(8)
    call = () -> train!(trained, optimizer, loader, batch, 2; show_progress = false)

    stats = @timed Base.invokelatest(call)
    history = stats.value
    emit(stdout, "TRAIN", :compile_time => stats.compile_time,
        :recompile_time => stats.recompile_time, :time => stats.time, :bytes => stats.bytes,
        :eltype => eltype(history), :finite => all(isfinite, history),
        :history => _no_space(join(round.(history; digits = 6), ",")))
    emit(stdout, "END", :status => "ok", :elapsed => round(time() - t0; digits = 1))
    exit(0)
end

# ---------------------------------------------------------------- reading a log back

"""
One run's log and the reports that carry its pid. `reports` is the reports directory; a report that
the parent did not attach but that carries the run's pid is attached here, which finds the file that
the crash reporter wrote after the parent had stopped waiting.
"""
function read_run(path::AbstractString; reports::AbstractString = REPORTS)
    lines = readlines(path; keep = true)
    header = findfirst(l -> startswith(l, "M8-RUN "), lines)
    status = findlast(l -> startswith(l, "M8-ARM-RUN "), lines)
    # A child that died before it printed `M8-RUN` (package load, Metal initialisation) leaves only
    # the parent's lines: its pid is then the one the parent took from the process handle, and its
    # arm is its directory's.
    header === nothing && status === nothing && return nothing
    source = header === nothing ? status : header
    arm = header === nothing ? basename(dirname(abspath(path))) :
          something(value_of(lines[header], "arm"), "-")
    pid = parse(Int, something(value_of(lines[source], "pid"), "0"))
    index = parse(Int, match(r"run-(\d+)\.log$", basename(path)).captures[1])

    endline = findlast(l -> startswith(l, "M8-END "), lines)
    endline = endline === nothing ? nothing : lines[endline]
    device = findfirst(l -> startswith(l, "M8-DEVICE functional="), lines)
    functional = device !== nothing &&
                 startswith(lines[device], "M8-DEVICE functional=true")
    envline = findfirst(l -> startswith(l, "M8-ENV "), lines)
    layer = envline === nothing ? "-" :
            replace(something(value_of(lines[envline], "MTL_DEBUG_LAYER"), "\"-\""), '"' => "")
    train = findfirst(l -> startswith(l, "M8-TRAIN "), lines)
    compile_time = train === nothing ? -1.0 :
                   parse(Float64, something(value_of(lines[train], "compile_time"), "-1"))
    finite = train !== nothing &&
             something(value_of(lines[train], "finite"), "false") == "true"
    exitcode = status === nothing ? 0 :
               parse(Int, something(value_of(lines[status], "exit"), "0"))
    reads = findlast(l -> startswith(l, "M8-READ "), lines)
    concurrent = reads !== nothing &&
                 something(value_of(lines[reads], "concurrent"), "false") == "true"
    overlap = overlap_evidence(lines)

    attached = NamedTuple{
        (:file, :pid, :match, :signature), Tuple{String, Int, Bool, Symbol}}[]
    # A report that existed before the run is not the run's, even when a reused pid matches it.
    seen = Set{String}()
    for line in lines
        startswith(line, "M8-IPS-BEFORE ") || continue
        union!(seen, filter(!isempty, split(something(value_of(line, "files"), ""), ",")))
    end
    for line in lines
        startswith(line, "M8-IPS ") || continue
        file = something(value_of(line, "file"), "-")
        push!(seen, file)
        other = parse(Int, something(value_of(line, "report"), "0"))
        # The signature is read from a report of this run's pid that still exists; any other
        # report, which macOS may have deleted, keeps the signature the parent logged.
        mine = other == pid
        src = report_path(path, index, file, reports)
        signature = mine && isfile(src) ? report_signature(src) :
                    Symbol(something(value_of(line, "signature"), "other"))
        push!(attached, (file = file, pid = other, match = mine, signature = signature))
    end
    # A report of this run's pid that appeared after the parent stopped waiting.
    for file in report_files(reports)
        file in seen && continue
        mine = report_pid(joinpath(reports, file)) == pid
        mine || continue
        push!(attached,
            (file = file, pid = pid, match = true,
                signature = report_signature(joinpath(reports, file))))
    end

    RunLog(
        arm, index, pid, exitcode, endline, functional, layer,
        any(l -> occursin("Metal API Validation Enabled", l), lines), concurrent, overlap,
        compile_time, finite, attached)
end

"""
Whether a session-arm log holds the records that the two halves write before the timed call can
finish: an `M8-TRAIN-START` line and an `M8-READ-START` line whose first read finished after the
trainer's `T0`, on another thread, and — when an `M8-TRAIN` line holds a `T1` — before it.
"""
function overlap_evidence(lines)
    train = findfirst(l -> startswith(l, "M8-TRAIN-START "), lines)
    read = findfirst(l -> startswith(l, "M8-READ-START "), lines)
    (train === nothing || read === nothing) && return false
    field(l, k) = parse(Int, something(value_of(l, k), "0"))
    t0 = field(lines[train], "T0")
    r0 = field(lines[read], "R0")
    done = findfirst(l -> startswith(l, "M8-TRAIN ") && value_of(l, "T1") !== nothing, lines)
    t1 = done === nothing ? typemax(Int) : field(lines[done], "T1")
    t0 > 0 && t0 < r0 < t1 &&
        field(lines[train], "trainer_thread") != field(lines[read], "reader_thread")
end

"The copied crash report of a run, or the one in the reports directory."
function report_path(log::AbstractString, index::Int, file::AbstractString, reports::AbstractString)
    copy = joinpath(dirname(log), "run-" * lpad(index, 3, '0') * "." * file)
    isfile(copy) ? copy : joinpath(reports, file)
end

"""
The rule, over what a log holds: `:abort` when a report of this run's own pid exists (in the
`lasympnet-concurrent` arm, only with a record of overlap), `:ok` when the
run trained a cold first call on a functional device with the debug layer set and shown active and
exited with 0 — and, in the `lasympnet-concurrent` arm, when its `M8-READ` line says that the reads
overlapped the training call — and `:void` with the reason for anything else.
"""
function verdict(run::RunLog)
    matched = filter(r -> r.match, run.reports)
    if !isempty(matched)
        # A death in the session arm is the arm's abort only with a record that the reads and the
        # training call overlapped; without one it is a recorded death outside the counted window.
        run.arm == "lasympnet-concurrent" && !run.overlap &&
            return (status = :void, signature = first(matched).signature,
                reason = "death with a report of this pid and no record of overlap")
        return (status = :abort, reason = "report of this pid",
            signature = first(matched).signature)
    end
    run.exit != 0 && return (status = :void, signature = :none,
        reason = "exit $(run.exit) with no report of this pid")
    !run.functional && return (status = :void, signature = :none,
        reason = "no functional Metal device")
    run.layer != "1" && return (status = :void, signature = :none,
        reason = "MTL_DEBUG_LAYER is $(run.layer)")
    run.validation || return (status = :void, signature = :none,
        reason = "the validation layer left no line")
    run.endline === nothing && return (status = :void, signature = :none,
        reason = "log has no M8-END line")
    occursin("status=ok", run.endline) || return (status = :void, signature = :none,
        reason = _no_space(run.endline))
    run.compile_time <= 0 && return (status = :void, signature = :none,
        reason = "compile_time is $(run.compile_time), a warm call")
    !run.finite &&
        return (status = :void, signature = :none, reason = "loss history is not finite")
    run.arm == "lasympnet-concurrent" && !run.concurrent &&
        return (status = :void, signature = :none,
            reason = "no M8-READ line says the reads overlapped the training call")
    (status = :ok, reason = "trained", signature = :none)
end

# ---------------------------------------------------------------- the parent

"The command that starts one run in a process of its own."
function child_cmd(arm::AbstractString, threads::Int)
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --threads=$(threads) $(@__FILE__) single --arm $(arm)`
    addenv(cmd, "MTL_DEBUG_LAYER" => "1")
end

"""
The exit code of a finished process as a shell gives it: `128 + n` for a death by signal `n`, whose
`exitcode` field is 0.
"""
exit_code(p::Base.Process) = p.termsignal == 0 ? p.exitcode : 128 + p.termsignal

"""
Run the counted runs `from:to` of one arm, each in a new process, and record each in its own log
beside the crash reports that carry its pid. The arm stops at the first report that shows the arm's
own abort.

The parent waits `CRASH_REPORTER_WAIT` seconds for the crash reporter only when the process did not
exit with 0; a death by a signal is recorded as `128 + n`, see `exit_code`. A report for a process
that exited cleanly cannot exist, and `audit` scans the reports directory again for every run's pid,
which finds a file that the reporter wrote late.
"""
function run_arm(arm::AbstractString, from::Int, to::Int, out::AbstractString;
        threads::Int = THREADS, reports::AbstractString = REPORTS)
    arm in ARMS || error("unknown arm $arm; the arms are $(ARMS)")
    arm == "lasympnet-concurrent" &&
        error("the concurrent arm needs the session's second ex; run it by hand, see the report")
    dir = joinpath(out, arm)
    mkpath(dir)
    for i in from:to
        logpath = joinpath(dir, "run-" * lpad(i, 3, '0') * ".log")
        isfile(logpath) && error("run $i of $arm has a log already: $logpath")
        before = report_files(reports)
        open(logpath, "w") do io
            emit(io, "IPS-BEFORE", :count => length(before), :files =>
                _no_space(join(before, ",")))
        end
        code = 0
        elapsed = 0.0
        pid = 0
        open(logpath, "a") do io
            t0 = time()
            p = run(
                pipeline(ignorestatus(child_cmd(arm, threads)); stdout = io,
                    stderr = io);
                wait = false)
            # The run's own pid, from the handle while the process runs: a handle of a process
            # that has exited no longer answers `getpid`, and a child that dies before it prints
            # `M8-RUN` prints none.
            pid = getpid(p)
            wait(p)
            code = exit_code(p)
            elapsed = time() - t0
        end
        code == 0 || sleep(CRASH_REPORTER_WAIT)
        after = report_files(reports)
        new = setdiff(after, before)
        matched = String[]
        open(logpath, "a") do io
            emit(io, "IPS-AFTER", :count => length(after), :new => length(new),
                :files => _no_space(join(new, ",")))
            for file in new
                other = report_pid(joinpath(reports, file))
                mine = other == pid
                mine && push!(matched, file)
                mine && cp(joinpath(reports, file),
                    joinpath(dir, "run-" * lpad(i, 3, '0') * "." * file); force = true)
                emit(io, "IPS", :run => i, :report => something(other, 0), :file => file,
                    :match => mine, :signature => report_signature(joinpath(reports, file)),
                    :symbols =>
                        _no_space(join(report_symbols(joinpath(reports, file)), "|")))
            end
            emit(io, "ARM-RUN", :run => i, :pid => pid, :exit => code,
                :elapsed => round(elapsed; digits = 1), :matched => length(matched))
        end
        isempty(matched) && continue
        signature = report_signature(joinpath(dir, "run-" * lpad(i, 3, '0') * "." *
                                                   first(matched)))
        expected = arm == "vpff" ? :malloc : :encoder
        signature == expected || continue
        open(logpath, "a") do io
            emit(io, "ARM-STOP", :arm => arm, :run => i, :reason => "reproduced",
                :signature => signature)
        end
        break
    end
    counts = audit_arm(joinpath(out, arm); reports = reports)
    println("== $(counts.arm): $(counts.state)")
    counts
end

# ---------------------------------------------------------------- the audit

"The runs of one arm's directory, in order."
function read_arm(dir::AbstractString; reports::AbstractString = REPORTS)
    runs = RunLog[]
    for f in sort!(filter(f -> occursin(r"^run-\d+\.log$", f), readdir(dir)))
        run = read_run(joinpath(dir, f); reports = reports)
        run === nothing || push!(runs, run)
    end
    runs
end

"""
The count of one arm, from its logs: how many runs it holds, how many trained, how many died with a
report of their own pid, how many are void and why, and whether a pid occurs twice.

The state is `reproduced` for a report that shows the arm's own abort, `no reproduction in
$(RUNS_PER_ARM) fresh processes` only at `RUNS_PER_ARM` clean runs, and `k of $(RUNS_PER_ARM), not
done` otherwise.
"""
function audit_arm(dir::AbstractString; reports::AbstractString = REPORTS)
    runs = read_arm(dir; reports = reports)
    seen = Set{Int}()
    clean = 0
    aborts = RunLog[]
    void = Pair{Int, String}[]
    reproduced = false
    for run in runs
        if run.pid in seen
            push!(void, run.index => "pid $(run.pid) is not fresh")
            continue
        end
        push!(seen, run.pid)
        v = verdict(run)
        if v.status === :ok
            clean += 1
        elseif v.status === :abort
            push!(aborts, run)
            (runs[1].arm == "vpff" ? :malloc : :encoder) == v.signature &&
                (reproduced = true)
        else
            push!(void, run.index => v.reason)
        end
    end
    state = reproduced ? "reproduced" :
            clean >= RUNS_PER_ARM ? "no reproduction in $(RUNS_PER_ARM) fresh processes" :
            "$(clean) of $(RUNS_PER_ARM), not done"
    (arm = basename(dir), runs = length(runs),
        clean = clean, aborts = aborts, void = void,
        state = state)
end

"Print the count of every arm under `out`, and return the counts."
function audit(out::AbstractString; reports::AbstractString = REPORTS)
    counts = Any[]
    for arm in ARMS
        dir = joinpath(out, arm)
        isdir(dir) || continue
        c = audit_arm(dir; reports = reports)
        push!(counts, c)
        println("== $(c.arm): $(c.state)")
        println("   runs $(c.runs), trained $(c.clean), aborts $(length(c.aborts)), void $(length(c.void))")
        for run in c.aborts, r in filter(r -> r.match, run.reports)

            println("   abort run $(run.index) pid $(run.pid): $(r.file) $(r.signature)")
        end
        for (index, reason) in c.void
            println("   void run $(index): $(reason)")
        end
    end
    counts
end

# ---------------------------------------------------------------- calibration

"""
The calibration process: not counted, and the same run as a counted one, so that a Bash process
settles whether it reaches Metal and whether the debug layer is visible before the first counted run.

It reports the device, its name, `Threads.nthreads()` and every package of the stack, and looks for
what the debug layer changes: the `Metal API Validation Enabled` line, or an `MTLDebug` class in the
device. A layer that is set but not shown to be active is reported as such.
"""
function calibrate(out::AbstractString; threads::Int = THREADS, reports::AbstractString = REPORTS)
    mkpath(out)
    path = joinpath(out, "calibration-" * string(round(Int, time())) * ".log")
    open(path, "w") do io
        run(
            pipeline(ignorestatus(child_cmd("lasympnet", threads)); stdout = io, stderr = io);
            wait = true)
    end
    text = read(path, String)
    visible = occursin("Metal API Validation Enabled", text) || occursin("MTLDebug", text)
    functional = occursin("M8-DEVICE functional=true", text)
    device = match(r"M8-DEVICE functional=true type=(\S+) name=(\S+)", text)
    emit(stdout, "CALIBRATE", :log => _no_space(path), :threads => threads,
        :functional => functional, :layer_visible => visible,
        :device => device === nothing ? "-" : device.captures[1] * "/" * device.captures[2])
    (path = path, visible = visible, functional = functional, text = text)
end

# ---------------------------------------------------------------- selftest

"""
The audit's own test, over logs that this function writes: a clean run, a run on a non-functional
device, a warm run, a run without the debug layer, a run whose validation layer left no line, a
reused pid, a death whose only new report carries another pid, a death with a report of its own
pid, a report of a run's pid that the parent did not attach, a report of a run's pid that existed
before the run, a child that died before `M8-RUN`, a foreign report that is gone, a session-arm run
whose reads did not overlap the training call, a session-arm death with and without the overlap
records, and an arm of fewer than `RUNS_PER_ARM` clean runs. Each must come back as the verdict it
is: a clean run, and a clean run beside an older report of its pid, as `:ok`, and every other case
never as `:ok`. It also checks the exit code of a process that died by a signal.
"""
function selftest()
    root = mktempdir()
    reports = joinpath(root, "reports")
    mkpath(reports)
    write(joinpath(reports, "julia-2026-01-01-000000.ips"), "{\n  \"pid\" : 999,\n}\n")
    body = """
    Metal API Validation Enabled
    M8-ENV MTL_DEBUG_LAYER="1" julia=1.13.3 nthreads=16 project=-
    M8-PKG name=Metal version=1.11.1 path=-
    M8-DEVICE functional=true type=MTLDevice name=AGXG16CDevice
    M8-TRAIN compile_time=1.0e9 recompile_time=0.0 time=1.0 bytes=1 eltype=Float32 finite=true history=0.5
    M8-END status=ok elapsed=1.0
    """
    function fixture(dir, index, pid, exit; arm = "lasympnet", layer = "1", device = true,
            compile = "1.0e9", endline = "M8-END status=ok", finite = "true",
            validation = true, extra = "")
        path = joinpath(dir, "run-" * lpad(index, 3, '0') * ".log")
        text = replace(body, "MTL_DEBUG_LAYER=\"1\"" => "MTL_DEBUG_LAYER=\"$layer\"",
            "compile_time=1.0e9" => "compile_time=$compile", "M8-END status=ok" => endline,
            "finite=true" => "finite=$finite")
        device ||
            (text = replace(text, "M8-DEVICE functional=true" => "M8-DEVICE functional=false"))
        validation || (text = replace(text, "Metal API Validation Enabled\n" => ""))
        open(path, "w") do io
            emit(io, "RUN", :arm => arm, :pid => pid,
                :time => 0.0, :concurrent => false)
            print(io, text)
            print(io, extra)
            emit(io, "ARM-RUN", :run => index, :pid => pid, :exit => exit, :elapsed => 1.0,
                :matched => 0)
        end
        path
    end
    failures = String[]
    checks = Ref(0)
    function check(name, got, want)
        checks[] += 1
        got === want || push!(failures, "$name: got $got, want $want")
    end
    verdict_of(dir, index) = verdict(read_run(
        joinpath(dir, "run-" * lpad(index, 3, '0') * ".log");
        reports = reports))

    one = joinpath(root, "one")
    mkpath(one)
    fixture(one, 1, 1001, 0)
    check("clean", verdict_of(one, 1).status, :ok)

    two = joinpath(root, "device")
    mkpath(two)
    fixture(two, 1, 1002, 0; device = false)
    check("null device", verdict_of(two, 1).status, :void)

    three = joinpath(root, "warm")
    mkpath(three)
    fixture(three, 1, 1003, 0; compile = "0.0")
    check("warm", verdict_of(three, 1).status, :void)

    four = joinpath(root, "layer")
    mkpath(four)
    fixture(four, 1, 1004, 0; layer = "")
    check("no layer", verdict_of(four, 1).status, :void)

    four_b = joinpath(root, "validation")
    mkpath(four_b)
    fixture(four_b, 1, 1010, 0; validation = false)
    check("no validation line", verdict_of(four_b, 1).status, :void)

    five = joinpath(root, "pid")
    mkpath(five)
    fixture(five, 1, 1005, 0)
    fixture(five, 2, 1005, 0)
    counts = audit_arm(five; reports = reports)
    check("reused pid, count", counts.clean, 1)
    check("reused pid, void", length(counts.void), 1)

    six = joinpath(root, "decoy")
    mkpath(six)
    # The parent's own `match=` field is not trusted: a report whose `pid` is not this run's is not
    # this run's, whatever the line says.
    fixture(six, 1, 1006, 134;
        extra = "M8-IPS-AFTER count=2 new=1 files=julia-2026-01-01-000000.ips\nM8-IPS run=1 report=999 file=julia-2026-01-01-000000.ips match=true signature=other symbols=-\n")
    check("decoy report", verdict_of(six, 1).status, :void)
    check("decoy is no abort", length(audit_arm(six; reports = reports).aborts), 0)

    seven = joinpath(root, "abort")
    mkpath(seven)
    write(joinpath(seven, "run-001.julia-2026-01-02-000000.ips"),
        "{\n  \"pid\" : 1007,\n  \"message\" : \"BUG IN CLIENT OF LIBMALLOC: memory corruption of free block\"\n}\n")
    fixture(seven, 1, 1007, 134; arm = "vpff",
        extra = "M8-IPS-AFTER count=2 new=1 files=julia-2026-01-02-000000.ips\nM8-IPS run=1 report=1007 file=julia-2026-01-02-000000.ips match=true signature=malloc symbols=-\n")
    check("abort", verdict_of(seven, 1).status, :abort)
    check("abort counted", length(audit_arm(seven; reports = reports).aborts), 1)
    check("malloc reproduces", audit_arm(seven; reports = reports).state, "reproduced")

    eleven = joinpath(root, "not finite")
    mkpath(eleven)
    fixture(eleven, 1, 1012, 0; finite = "false")
    check("not finite", verdict_of(eleven, 1).status, :void)

    twelve = joinpath(root, "status")
    mkpath(twelve)
    fixture(twelve, 1, 1013, 0; endline = "M8-END status=error reason=-")
    check("error status", verdict_of(twelve, 1).status, :void)

    eight = joinpath(root, "short")
    mkpath(eight)
    for i in 1:3
        fixture(eight, i, 1100 + i, 0)
    end
    check("short arm", audit_arm(eight; reports = reports).state,
        "3 of $(RUNS_PER_ARM), not done")

    # A report of a run's pid that the parent did not attach still counts, which is the late file
    # that the crash reporter writes after the wait.
    nine = joinpath(root, "late")
    mkpath(nine)
    fixture(nine, 1, 1009, 134)
    write(joinpath(reports, "julia-2026-01-03-000000.ips"),
        "{\n  \"pid\" : 1009,\n  \"message\" : \"BUG IN CLIENT OF LIBMALLOC: memory corruption of free block\"\n}\n")
    check("late report", verdict_of(nine, 1).status, :abort)

    # A command-encoder fault of a `lasympnet` run reproduces that arm's abort, which no count of
    # aborts says: the signature is what decides it.
    ten = joinpath(root, "encoder")
    mkpath(ten)
    write(joinpath(ten, "run-001.julia-2026-01-04-000000.ips"),
        "{\n  \"pid\" : 1011,\n  \"symbols\" : \"MTLReleaseAssertionFailure -[IOGPUMetalCommandBuffer setCurrentCommandEncoder:]\"\n}\n")
    fixture(ten, 1, 1011, 134;
        extra = "M8-IPS-AFTER count=2 new=1 files=julia-2026-01-04-000000.ips\nM8-IPS run=1 report=1011 file=julia-2026-01-04-000000.ips match=true signature=encoder symbols=-\n")
    check("encoder signature", verdict_of(ten, 1).signature, :encoder)
    check("encoder reproduces", audit_arm(ten; reports = reports).state, "reproduced")

    # A report that was in the directory before the run is not the run's, even with its pid.
    thirteen = joinpath(root, "reused report pid")
    mkpath(thirteen)
    write(joinpath(reports, "julia-2026-01-05-000000.ips"),
        "{\n  \"pid\" : 1014,\n  \"message\" : \"BUG IN CLIENT OF LIBMALLOC: memory corruption of free block\"\n}\n")
    fixture(thirteen, 1, 1014, 0;
        extra = "M8-IPS-BEFORE count=1 files=julia-2026-01-05-000000.ips\n")
    check("report from before the run", verdict_of(thirteen, 1).status, :ok)

    # The session arm counts a clean run only when its reads overlapped the training call.
    fourteen = joinpath(root, "concurrent")
    mkpath(fourteen)
    fixture(fourteen, 1, 1015, 0; arm = "lasympnet-concurrent")
    fixture(fourteen, 2, 1016, 0; arm = "lasympnet-concurrent",
        extra = "M8-READ R0=2 R1=3 T0=1 T1=4 reads=9 reader_thread=2 trainer_thread=1 concurrent=false\n")
    fixture(fourteen, 3, 1017, 0; arm = "lasympnet-concurrent",
        extra = "M8-READ R0=2 R1=3 T0=1 T1=4 reads=9 reader_thread=2 trainer_thread=1 concurrent=true\n")
    check("no read line", verdict_of(fourteen, 1).status, :void)
    check("reads did not overlap", verdict_of(fourteen, 2).status, :void)
    check("reads overlapped", verdict_of(fourteen, 3).status, :ok)

    # A child that died before `M8-RUN` leaves only the parent's lines. Its pid is the parent's, and
    # the run is void, or an abort when a report of that pid exists.
    fifteen = joinpath(root, "before M8-RUN")
    mkpath(fifteen)
    for (i, pid) in ((1, 1020), (2, 1021))
        write(joinpath(fifteen, "run-" * lpad(i, 3, '0') * ".log"),
            "M8-IPS-BEFORE count=0 files=\nM8-IPS-AFTER count=0 new=0 files=\nM8-ARM-RUN run=$i pid=$pid exit=134 elapsed=1.0 matched=0\n")
    end
    write(joinpath(reports, "julia-2026-01-06-000000.ips"),
        "{\n  \"pid\" : 1021,\n  \"message\" : \"BUG IN CLIENT OF LIBMALLOC: memory corruption of free block\"\n}\n")
    check("death before M8-RUN", verdict_of(fifteen, 1).status, :void)
    check("death before M8-RUN, pid", read_run(joinpath(fifteen, "run-001.log"); reports).pid, 1020)
    check("death before M8-RUN with a report", verdict_of(fifteen, 2).status, :abort)
    check("death before M8-RUN, counted", audit_arm(fifteen; reports = reports).runs, 2)

    # A foreign report that the parent logged and macOS has since deleted is not read.
    sixteen = joinpath(root, "foreign report gone")
    mkpath(sixteen)
    fixture(sixteen, 1, 1022, 0;
        extra = "M8-IPS-AFTER count=1 new=1 files=julia-gone.ips\nM8-IPS run=1 report=7777 file=julia-gone.ips match=false signature=other symbols=-\n")
    check("foreign report gone", verdict_of(sixteen, 1).status, :ok)

    # In the session arm, a death with a report of the run's pid is the arm's abort only with the
    # records that the two halves write before the timed call ends.
    seventeen = joinpath(root, "session death")
    mkpath(seventeen)
    for (i, pid) in ((1, 1023), (2, 1024))
        write(
            joinpath(seventeen, "run-" * lpad(i, 3, '0') * ".julia-2026-01-07-00000$i.ips"),
            "{\n  \"pid\" : $pid,\n  \"symbols\" : \"endComputePass Metal\"\n}\n")
    end
    died(i, pid, extra) = fixture(seventeen, i, pid, 139; arm = "lasympnet-concurrent",
        extra = extra *
                "M8-IPS-AFTER count=1 new=1 files=julia-2026-01-07-00000$i.ips\nM8-IPS run=$i report=$pid file=julia-2026-01-07-00000$i.ips match=true signature=encoder symbols=-\n")
    died(1, 1023, "")
    died(2, 1024,
        "M8-TRAIN-START T0=10 trainer_thread=1\nM8-READ-START R0=20 T0=10 reader_thread=2 trainer_thread=1\n")
    check("session death without overlap record", verdict_of(seventeen, 1).status, :void)
    check("session death without overlap record, signature",
        verdict_of(seventeen, 1).signature, :encoder)
    check("session death with overlap record", verdict_of(seventeen, 2).status, :abort)
    rm(joinpath(seventeen, "run-002.log"))
    check("no overlap record, not reproduced",
        audit_arm(seventeen; reports = reports).state, "0 of $(RUNS_PER_ARM), not done")
    died(2, 1024,
        "M8-TRAIN-START T0=10 trainer_thread=1\nM8-READ-START R0=20 T0=10 reader_thread=2 trainer_thread=1\n")
    check("overlap record, reproduced",
        audit_arm(seventeen; reports = reports).state, "reproduced")

    # A death by a signal leaves `exitcode` 0; the parent records `128 + n`.
    check("signal exit code", exit_code(run(ignorestatus(`sh -c "kill -ABRT \$\$"`))), 134)

    rm(root; recursive = true, force = true)
    isempty(failures) && (println("selftest: PASS, $(checks[]) checks"); return 0)
    foreach(f -> println("selftest: FAIL ", f), failures)
    1
end

# ---------------------------------------------------------------- entry

function main(args)
    isempty(args) &&
        error("usage: metal_abort_runs.jl calibrate | run | audit | selftest [options]")
    mode = args[1]
    options = Dict{String, String}()
    i = 2
    while i <= length(args)
        arg = args[i]
        startswith(arg, "--") || error("unexpected argument $arg")
        key = arg[3:end]
        i + 1 <= length(args) || error("--$key needs a value")
        options[key] = args[i + 1]
        i += 2
    end
    out = get(options, "out", joinpath(pwd(), "m8-runs"))
    threads = parse(Int, get(options, "threads", string(THREADS)))
    reports = get(options, "reports", REPORTS)
    if mode == "single"
        # `using` is evaluated here, at top level of the module, and `single` runs in the world it
        # creates: a function that names a package the module did not yet import is a world-age
        # error.
        @eval using Metal, GeometricMachineLearning, GeometricOptimizers,
                    KernelAbstractions,
                    Zygote, Random, LinearAlgebra
        Base.invokelatest(single, get(options, "arm", "lasympnet"))
    elseif mode == "calibrate"
        calibrate(out; threads = threads, reports = reports)
    elseif mode == "run"
        from, to = split(get(options, "runs", "1:$(RUNS_PER_ARM)"), ":")
        run_arm(get(options, "arm", "lasympnet"), parse(Int, from), parse(Int, to), out;
            threads = threads, reports = reports)
    elseif mode == "audit"
        audit(out; reports = reports)
    elseif mode == "selftest"
        exit(selftest())
    else
        error("unknown mode $mode")
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
