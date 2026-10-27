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
#     ex: training()                                                   # returns a job id at 30 s
#     ex: reader()                                                     # sent at once, while the
#                                                                      # training call still runs
#
# The training `ex` must still be running when the tool returns the job id, which it does after 30 s;
# a `training()` call that returns a value before that leaves nothing to overlap, and the run is
# void. `T0` is stamped before the timed call and `T1` after it, and the reader loops on
# `sum(rand(Float32, 10^6))` and `yield()` from `T0` until `T1` is set.
#
# A run is concurrent when `T0 < R0 < T1`, at least one read finished before `T1`, and the reader's
# `threadid()` differs from the trainer's. Any other run is void and is replaced, and three void runs
# in a row stop the arm.
#
# ## The log of a run
#
# Each run's `M8-` lines go into `<out>/lasympnet-concurrent/run-<NNN>.log`, which is the layout
# `metal_abort_runs.jl audit` reads for the other two arms, so that the three arms are counted by one
# rule. The session writes the lines that begin `M8-RUN`, `M8-ENV`, `M8-PKG`, `M8-DEVICE`,
# `M8-FORWARD`, `M8-PULLBACK`, `M8-TRAIN`, `M8-READ` and `M8-END`; the person running it adds the
# two lines that a session cannot write:
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

"""
The training half of one run: the device and the stack, a forward pass and a pullback on one
network, then the first training call of a new network between the stamps `M8_T0` and `M8_T1`.

It ends by printing the stamps `T0` and `T1` and this thread's `threadid()` in its `M8-TRAIN` line,
which the reading half's `M8-READ` line is read against to say whether the two overlapped.
"""
function training(arm::AbstractString = "lasympnet")
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

    global M8_T1 = 0
    global M8_TRAINER = Threads.threadid()
    global M8_T0 = time_ns()
    stats = @timed Base.invokelatest(call)
    global M8_T1 = time_ns()
    history = stats.value
    emit(stdout, "TRAIN", :compile_time => stats.compile_time,
        :recompile_time => stats.recompile_time, :time => stats.time, :bytes => stats.bytes,
        :eltype => eltype(history), :finite => all(isfinite, history),
        :history => _no_space(join(round.(history; digits = 6), ",")),
        :trainer_thread => Threads.threadid(), :T0 => M8_T0, :T1 => M8_T1)
    emit(stdout, "END", :status => "ok", :elapsed => round(time() - t0; digits = 1))
    (T0 = M8_T0, T1 = M8_T1, trainer = Threads.threadid(), history = history)
end

"""
The reading half: wait for `M8_T0`, then read a host `Vector{Float32}` of 10⁶ entries and `yield()`
in a loop until `M8_T1` is set. Return `R0`, `R1`, the number of reads and this thread's id.

`R0` is the instant the first read finished, which is what edge 5 asks for: a run is concurrent when
`M8_T0 < R0 < M8_T1`, at least one read finished before `T1`, and the two `threadid()`s differ.
"""
function reader(n::Int = 10^6)
    v = rand(Float32, n)
    deadline = time() + 600
    while !isdefined(Main, :M8_T0) || M8_T0 == 0
        time() > deadline && error("no T0 after 600 s")
        yield()
    end
    R0 = 0
    reads = 0
    while !isdefined(Main, :M8_T1) || M8_T1 == 0
        sum(v)
        yield()
        reads += 1
        R0 == 0 && (R0 = time_ns())
        time() > deadline && error("no T1 after 600 s")
    end
    R1 = time_ns()
    T0, T1 = M8_T0, M8_T1
    trainer = isdefined(Main, :M8_TRAINER) ? M8_TRAINER : 0
    emit(stdout, "READ", :R0 => R0, :R1 => R1, :T0 => T0, :T1 => T1, :reads => reads,
        :reader_thread => Threads.threadid(), :trainer_thread => trainer,
        :concurrent => T0 < R0 < T1 && trainer != Threads.threadid())
    (R0 = R0, R1 = R1, T0 = T0, T1 = T1, reads = reads, reader = Threads.threadid(),
        trainer = trainer)
end
