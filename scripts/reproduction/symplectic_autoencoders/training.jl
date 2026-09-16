"""
TODO

- Implement variational autoencoder!!! Up to now the variational property has not been included.
- Make the computation of the reduction error automatic! (this has to be done for many values!)
- Also try using analytical data!!!
- At the moment you are training every network 4 times. 好傻，笨蛋。
"""

using GeometricMachineLearning
using HDF5
using CUDA
using GeometricIntegrators
using CairoMakie

# `smoke_size(full, smoke)` returns the second when GML_SMOKE is set, which is how the CI
# job runs this script end to end in seconds. See scripts/README.md.
include("../../utilities/smoke.jl")

include("../../utilities/vector_fields.jl")
include("../../utilities/initial_condition.jl")

T = Float64
n_epochs = smoke_size(100, 2)
# The batch size the hand-rolled loop this replaced never named: it read `dl.batch_size`, which a
# `DataLoader` has not carried since batching moved into `Batch`.
batch_size = smoke_size(512, 16)
n_range = 2:1:smoke_size(15, 3)
μ_range = (T(0.51), T(0.625), T(0.55), T(0.47))
# The learning rate is the `Optimizer`'s `step_size`, not the method's: `Adam` lost the `η` field
# it never applied to the direction, and `β₁`, `β₂` and `δ` became keywords so that the old
# positional call fails instead of silently reading `η` as `β₁`.
opt = AdamOptimizer(T; β₁ = T(0.9), β₂ = T(0.99), δ = T(1.0e-8))
const step_size = T(0.001)
# The retraction belongs to the `Optimizer` now, not to the layer: `PSDLayer(M, N)` takes no
# keyword.
retraction = Cayley()

# The snapshot matrix is what `integration.jl` writes, into whichever directory it is run from.
# Stating the dependency here rather than relying on the caller having run it first.
isfile("snapshot_matrix.h5") || include("integration.jl")

function gpu_backend()
    data = h5open("snapshot_matrix.h5", "r") do file
        read(file, "data")
    end
    n_params = h5open("snapshot_matrix.h5", "r") do file
        read(file, "n_params")
    end
    (CUDABackend(), data |> cu, n_params)
end

function cpu_backend()
    data = h5open("snapshot_matrix.h5", "r") do file
        read(file, "data")
    end
    n_params = h5open("snapshot_matrix.h5", "r") do file
        read(file, "n_params")
    end
    (CPU(), data, n_params)
end

backend, data, n_params = try
    gpu_backend()
catch
    cpu_backend()
end

dl = DataLoader(data)
n_time_steps = size(data, 2) / n_params
N = size(data, 1)÷2

# Both reductions are neural networks now, because that is what `HRedSys` takes: every one of its
# constructors wants a `NeuralNetwork{<:SymplecticEncoder}` and a `NeuralNetwork{<:SymplecticDecoder}`,
# where this script used to build a pair of closures. `PSDArch` is the proper orthogonal
# decomposition this script used to compute with a hand-written `svd`, and `solve!` is how it is
# fitted -- there is nothing to train.
function get_psd_encoder_decoder(; n = 5)
    psd_nn = NeuralNetwork(PSDArch(2 * N, 2 * n), backend, T)
    solve!(psd_nn, dl)
    psd_nn
end

# `SymplecticAutoencoder` is the architecture the hand-built encoder/decoder chain was spelling out:
# gradient layers around a `PSDLayer` that changes dimension, in both directions. Building it as an
# architecture rather than a bare `Chain` is what makes `encoder(nn)` and `decoder(nn)` available,
# and those are what `HRedSys` needs.
#
# Training is `DataLoader` + `Batch` + `Optimizer`. What stood here was a hand-rolled loop over
# `redraw_batch!(dl)`, counting its own iterations from `dl.batch_size`; a `DataLoader` carries
# neither, because the batch is a `Batch` and the loop is the `Optimizer` functor. The loop also
# wrapped both the pullback and the optimization step in `try … catch; continue`, so a training run
# in which every single step failed was indistinguishable from one that worked.
function get_nn_encoder_decoder(; n = 5, n_epochs = 500, opt = opt, T = T)
    sae_arch = SymplecticAutoencoder(
        2 * N, 2 * n; n_encoder_blocks = 4, n_decoder_blocks = 4,
        n_encoder_layers = 2, n_decoder_layers = 2)
    sae_nn = NeuralNetwork(sae_arch, backend, T)

    optimizer_instance = Optimizer(opt, sae_nn; step_size = step_size, retraction = retraction)
    optimizer_instance(sae_nn, dl, Batch(batch_size), n_epochs)

    sae_nn
end

# `ReducedSystem` is `HRedSys`, and the reduced vector field is no longer assembled by hand:
# `reduced_vector_field_from_full_explicit_vector_field` is gone, and the constructor below derives
# it from the problem and the decoder. `Symplectic()` is gone with it — `HRedSys` is the Hamiltonian
# reduced system by construction. The problem is the same wave equation `integration.jl` builds.
function get_reduced_model(autoencoder; μ_val = 0.51, Ñ = (N-2),
        n_time_steps = n_time_steps, integrator = ImplicitMidpoint())
    params = (μ = μ_val, Ñ = Ñ, Δx = T(1/(Ñ-1)))
    timestep = T(1/(n_time_steps-1))
    timespan = (T(0), T(1))
    ics_offset = get_initial_condition(μ_val, Ñ)
    ics = (q = ics_offset.q.parent, p = ics_offset.p.parent)
    problem = HODEProblem(
        v_f_hamiltonian(params)..., parameters = params, timespan, timestep, ics)
    HRedSys(problem, encoder(autoencoder), decoder(autoencoder); integrator = integrator)
end

function _cpu_convert(ps::Tuple)
    output = ()
    for elem in ps
        output = (output..., _cpu_convert(elem))
    end
    output
end

function _cpu_convert(ps::NamedTuple)
    output = ()
    for elem in ps
        output = (output..., _cpu_convert(elem))
    end
    NamedTuple{keys(ps)}(output)
end

_cpu_convert(A::AbstractArray) = Array(A)

_cpu_convert(Y::StiefelManifold) = StiefelManifold(_cpu_convert(Y.A))

# Neither this nor `plot_comparison_for_reconstructed_trajectories` below is called anywhere in this
# script, and neither was before. They are carried over rather than deleted, with the two renamed
# integration entry points applied so that the file names nothing that no longer exists.
function get_reconstructed_trajectories(psd_rs, nn_rs)
    psd_time_series = integrate_reduced_system(psd_rs)
    nn_time_series = integrate_reduced_system(nn_rs)
    for t in axes(psd_time_series.q, 1)
        psd_time_series.q[t] = psd_rs.decoder(psd_time_series.q[t])
        nn_time_series.q[t] = nn_rs.decoder(nn_time_series.q[t])
    end
    (psd = psd_time_series, nn = nn_time_series, full = integrate_full_system(psd_rs))
end

function plot_comparison_for_reconstructed_trajectories(trajectories, t_step = 0)
    n_t_steps = length(trajectories.psd.t)
    t_step_index = Int(ceil(n_t_steps*t_step))
    N = length(trajectories.full.q[t_step_index])÷2
    plot_object = Figure()
    ax = Axis(plot_object[1, 1])
    lines!(ax, trajectories.full.q[t_step_index][1:N]; label = "Numerical solution")
    lines!(ax, trajectories.psd.q[t_step_index][1:N]; label = "PSD")
    lines!(ax, trajectories.nn.q[t_step_index][1:N]; label = "NN")
    axislegend(ax)
    mkpath("plots")
    CairoMakie.save("plots/comparison_for_time_step_"*string(t_step)*".png", plot_object)
end

data_cpu = _cpu_convert(data)

function get_enocders_decoders(n_range)
    encoders_decoders = NamedTuple()
    for n in n_range
        encoders_decoders_current = (
            nn = get_nn_encoder_decoder(n = n, n_epochs = n_epochs),
            psd = get_psd_encoder_decoder(n = n))
        encoders_decoders = NamedTuple{(keys(encoders_decoders)..., Symbol("n"*string(n)))}((
            values(encoders_decoders)..., encoders_decoders_current))
    end
    encoders_decoders
end

encoders_decoders = get_enocders_decoders(n_range)

μ_errors = NamedTuple()
for μ_test_val in μ_range
    errors = NamedTuple()
    # The full solution does not depend on the reduction, so it is integrated once per `μ`. It used
    # to come from a `ReducedSystem` built with `nothing` for both encoder and decoder; `HRedSys`
    # takes neural networks, so the first reduced system of the sweep supplies it instead.
    sol_full = nothing
    for n in n_range
        current_n_identifier = Symbol("n"*string(n))
        encoders_decoders_current = encoders_decoders[current_n_identifier]

        psd_rs = get_reduced_model(
            encoders_decoders_current.psd; μ_val = μ_test_val, Ñ = (N-2))
        nn_rs = get_reduced_model(
            encoders_decoders_current.nn; μ_val = μ_test_val, Ñ = (N-2))

        sol_full === nothing && (sol_full = integrate_full_system(psd_rs))

        reduction_errors = (psd = reduction_error(psd_rs, sol_full),
            nn = reduction_error(nn_rs, sol_full))
        projection_errors = (psd = projection_error(psd_rs, sol_full),
            nn = projection_error(nn_rs, sol_full))
        temp_errors = (
            reduction_error = reduction_errors, projection_error = projection_errors)
        errors = NamedTuple{(keys(errors)..., current_n_identifier)}((
            values(errors)..., temp_errors))
    end

    global μ_errors = NamedTuple{(keys(μ_errors)..., Symbol("μ"*string(μ_test_val)))}((
        values(μ_errors)..., errors))
end

function plot_projection_reduction_errors(μ_errors)
    number_errors = length(μ_errors[1])
    n_vals = zeros(Int, number_errors)
    nn_projection_vals = zeros(number_errors)
    nn_reduction_vals = zeros(number_errors)
    psd_projection_vals = zeros(number_errors)
    psd_reduction_vals = zeros(number_errors)
    for μ_key in keys(μ_errors)
        μ = string(μ_key)
        it = 0
        for n_key in keys(μ_errors[μ_key])
            it += 1
            n = parse(Int, string(n_key)[2:end])
            nn_projection_val = μ_errors[μ_key][n_key].projection_error.nn
            nn_reduction_val = μ_errors[μ_key][n_key].reduction_error.nn
            psd_projection_val = μ_errors[μ_key][n_key].projection_error.psd
            psd_reduction_val = μ_errors[μ_key][n_key].reduction_error.psd
            n_vals[it] = n
            nn_projection_vals[it] = nn_projection_val
            nn_reduction_vals[it] = nn_reduction_val
            psd_projection_vals[it] = psd_projection_val
            psd_reduction_vals[it] = psd_reduction_val
        end
        plot_object = Figure()
        ax = Axis(plot_object[1, 1]; limits = (nothing, (0, 1)))
        scatter!(ax, n_vals, psd_projection_vals; color = Makie.wong_colors()[2],
            marker = :cross, label = "PSD projection")
        scatter!(ax, n_vals, psd_reduction_vals;
            color = Makie.wong_colors()[2], label = "PSD reduction")

        scatter!(ax, n_vals, nn_projection_vals; color = Makie.wong_colors()[3],
            marker = :cross, label = "NN projection")
        scatter!(ax, n_vals, nn_reduction_vals;
            color = Makie.wong_colors()[3], label = "NN reduction")
        axislegend(ax)
        mkpath("plots")
        CairoMakie.save("plots/v3mu"*μ[3:end]*".png", plot_object)
    end
end

plot_projection_reduction_errors(μ_errors)
