# The `LASympNet(4)` arm with a concurrent evaluation: the training half and the reading half.
#
# `metal_abort_runs.jl` runs the two arms that need no second evaluation, one OS process per run. It
# cannot run this arm: the concurrency is a second *request* to one session, and a session's request
# is not something a script can make. The abort it asks about happened exactly here — a Kaimon `ex`
# that only read a Julia vector, while the first Adam call of `LASympNet(4)` ran in the same
# session.
#
# ## The run, for each counted run
#
# A private session on the scratch environment, restarted before every run, so that each run is a
# process of its own and no earlier run's state is left in it:
#
#     start_session(project_path = "<scratch env>", private = true)     # once
#     manage_repl(command = "restart", session = key)                  # before each run
#     ex: ENV["MTL_DEBUG_LAYER"] = "1"                                 # before Metal is loaded
#     ex: include("<tree>/scripts/utilities/metal_abort_session.jl")   # once per process
#     ex: training(log = "<out>/lasympnet-concurrent/run-<NNN>.log")   # returns a job id at 30 s
#     ex: reader(log = "<out>/lasympnet-concurrent/run-<NNN>.log")     # sent at once, while the
#                                                                      # training call still runs
#
# The training `ex` must still be running when the tool returns the job id, which it does after 30 s;
# a `training` call that returns a value before that leaves nothing to overlap, and the run is
# void. `T0` is stamped before the timed call and `T1` after it, and the reader loops on
# `sum(rand(Float32, 10^6))` and `yield()` from `T0` until `T1` is set.
#
# A run is concurrent when `T0 < R0 < T1`, at least one read finished before `T1`, and the reader's
# `threadid()` differs from the trainer's. Any other run is void and is replaced, and three void runs
# in a row stop the arm.
#
# The two halves append a record to the run's log file before the timed call can finish:
# `M8-TRAIN-START` with `T0` at the start of the call, and `M8-READ-START` with `R0` when the first
# read has finished. They go to the file and not to `stdout`, because a session's output is lost
# when the session dies. A run that dies leaves them, and `metal_abort_runs.jl audit` counts a death
# in this arm as the arm's abort only when the two records show `T0 < R0` on different threads.
#
# ## The log of a run
#
# Each run's `M8-` lines go into `<out>/lasympnet-concurrent/run-<NNN>.log`, which is the layout
# `metal_abort_runs.jl audit` reads for the other two arms, so that the three arms are counted by one
# rule. The session writes the lines that begin `M8-RUN`, `M8-ENV`, `M8-PKG`, `M8-DEVICE`,
# `M8-FORWARD`, `M8-PULLBACK`, `M8-TRAIN-START`, `M8-READ-START`, `M8-TRAIN`, `M8-READ` and
# `M8-END`, of which the two `-START` records go straight into the file and the others are the
# session's output, which the person running it appends to the same file. The session output also
# holds the `Metal API Validation Enabled` line that Metal prints when it loads with
# `MTL_DEBUG_LAYER=1`; `audit` requires it, and a log without it is void. The person adds the two
# lines that a session cannot write:
#
#     M8-IPS run=<N> report=<pid> file=<julia-*.ips> match=<true|false> signature=<encoder|malloc|other> symbols=<a|b|c>
#     M8-ARM-RUN run=<N> pid=<pid> exit=<code> elapsed=<s> matched=<n> concurrent=<true|false>
#
# and copies each report whose `"pid"` is the run's into the same directory as
# `run-<NNN>.<julia-*.ips>`.

include(joinpath(@__DIR__, "metal_abort_runs.jl"))

# The stack, loaded here so that the session's only preparation is the `MTL_DEBUG_LAYER` of an
# earlier `ex`: the layer has to be set before Metal is loaded.
using Metal, GeometricMachineLearning, GeometricOptimizers, KernelAbstractions, Zygote,
      Random,
      LinearAlgebra

# The stamps and the trainer's thread, shared by the two halves. Every read and write, and every
# append to the run's log, holds `M8_LOCK`, and the state exists from the `include`, before either
# request is sent.
const M8_LOCK = ReentrantLock()
const M8_STATE = Dict{Symbol, Int}(:t0 => 0, :t1 => 0, :trainer => 0)
state_set!(kv::Pair{Symbol}...) = @lock M8_LOCK merge!(M8_STATE, Dict(kv))
state_get(k::Symbol) = @lock M8_LOCK M8_STATE[k]
state_snapshot() = @lock M8_LOCK copy(M8_STATE)

"""
One `M8-` record appended to the run's log file and closed, so that it reaches the kernel and
survives a death of the process. A session's `flush(stdout)` does not: inside a Kaimon `ex` the
output goes through a channel, and the reading half's `ex` has no terminal at all.

The append holds `M8_LOCK`: `open(log, "a")` seeks to the end once and does not open with
`O_APPEND`, so two tasks that append at once can write at the same offset and lose a record.
"""
function emit_record(log::AbstractString, tag::AbstractString, fields::Pair...)
    @lock M8_LOCK open(io -> emit(io, tag, fields...), log, "a")
end

"""
The training half of one run: the device and the stack, a forward pass and a pullback on one
network, then the first training call of a new network between the stamps `:t0` and `:t1` of
`M8_STATE`.

It appends an `M8-TRAIN-START` line with `T0` to the file `log` before the timed call can finish,
and ends by printing the stamps `T0` and `T1` and this thread's `threadid()` in its `M8-TRAIN`
line. The reading half's `M8-READ-START` and `M8-READ` lines are read against them to say whether
the two overlapped.
"""
function training(arm::AbstractString = "lasympnet"; log::AbstractString)
    t0 = time()
    layer = get(ENV, "MTL_DEBUG_LAYER", "-")
    emit(stdout, "RUN", :arm => arm * "-concurrent", :pid => getpid(), :time => t0,
        :concurrent => true)
    emit(stdout, "ENV", :MTL_DEBUG_LAYER => "\"$layer\"", :julia => VERSION,
        :nthreads => Threads.nthreads(), :project => _no_space(Base.active_project()))
    print_stack(stdout)
    if layer != "1"
        emit(stdout, "END", :status => "error", :reason => "MTL_DEBUG_LAYER_is_not_1")
        return nothing
    end
    if !Metal.functional()
        emit(stdout, "DEVICE", :functional => false)
        emit(stdout, "END", :status => "error", :reason => "no_functional_Metal_device")
        return nothing
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
    nn = NeuralNetwork(LASympNet(4), MetalBackend(), Float32)
    y = nn(X)
    emit(stdout, "FORWARD", :eltype => eltype(y), :size => _no_space(size(y)))
    loss(ps) = sum(abs2, nn.model(X, ps))
    value, back = Zygote.pullback(loss, GeometricMachineLearning.params(nn))
    pullback = back(one(value))
    emit(stdout, "PULLBACK", :value => value, :leaves => length(pullback))

    trained = NeuralNetwork(LASympNet(4), MetalBackend(), Float32)
    optimizer = TrainingOptimizer(GeometricMachineLearning.params(trained); algorithm = Adam())
    loader = DataLoader(X; autoencoder = false, suppress_info = true)
    batch = Batch(8)
    call = () -> train!(trained, optimizer, loader, batch, 2; show_progress = false)

    trainer = Threads.threadid()
    state_set!(:t1 => 0, :trainer => trainer, :t0 => time_ns())
    # Flushed before the timed call can finish, so that a crash leaves the record of its start.
    emit_record(log, "TRAIN-START", :T0 => state_get(:t0), :trainer_thread => trainer)
    stats = @timed Base.invokelatest(call)
    state_set!(:t1 => time_ns())
    s = state_snapshot()
    history = stats.value
    emit(stdout, "TRAIN", :compile_time => stats.compile_time,
        :recompile_time => stats.recompile_time, :time => stats.time, :bytes => stats.bytes,
        :eltype => eltype(history), :finite => all(isfinite, history),
        :history => _no_space(join(round.(history; digits = 6), ",")),
        :trainer_thread => Threads.threadid(), :T0 => s[:t0], :T1 => s[:t1])
    emit(stdout, "END", :status => "ok", :elapsed => round(time() - t0; digits = 1))
    (T0 = s[:t0], T1 = s[:t1], trainer = Threads.threadid(), history = history)
end

"""
The reading half: wait for `:t0` of `M8_STATE`, then read a host `Vector{Float32}` of 10⁶ entries
and `yield()` in a loop until `:t1` is set. Return `R0`, `R1`, the number of reads and this thread's
id.

`R0` is the instant the first read finished, which the `M8-READ-START` line records in the file
`log`: a run is concurrent when `T0 < R0 < T1`, at least one read finished before `T1`, and the two
`threadid()`s differ.
"""
function reader(n::Int = 10^6; log::AbstractString)
    v = rand(Float32, n)
    deadline = time() + 600
    while state_get(:t0) == 0
        time() > deadline && error("no T0 after 600 s")
        yield()
    end
    R0 = 0
    reader_thread = 0
    reads = 0
    while state_get(:t1) == 0
        sum(v)
        if R0 == 0
            # The thread of the first read, sampled with no `yield` between the read and the stamp:
            # a task may resume on another thread after a `yield`.
            R0 = time_ns()
            reader_thread = Threads.threadid()
            # Flushed at the first finished read, so that a crash leaves the record of the overlap.
            s = state_snapshot()
            emit_record(log, "READ-START", :R0 => R0, :T0 => s[:t0],
                :reader_thread => reader_thread, :trainer_thread => s[:trainer])
        end
        yield()
        reads += 1
        time() > deadline && error("no T1 after 600 s")
    end
    R1 = time_ns()
    s = state_snapshot()
    T0, T1, trainer = s[:t0], s[:t1], s[:trainer]
    emit(stdout, "READ", :R0 => R0, :R1 => R1, :T0 => T0, :T1 => T1, :reads => reads,
        :reader_thread => reader_thread, :trainer_thread => trainer,
        :concurrent => T0 < R0 < T1 && trainer != reader_thread)
    (R0 = R0, R1 = R1, T0 = T0, T1 = T1, reads = reads, reader = reader_thread,
        trainer = trainer)
end
