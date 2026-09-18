# Train a `ForcedSympNet` on the damped harmonic oscillator and save the trained parameters.
#
# Damping is what makes this a forced problem: the friction term enters the `ṗ` equation, which is
# exactly what a `ForcingLayer` adds. There are no parameters of the *system* here, so this is a
# plain `DataLoader` rather than a `ParametricDataLoader`.
using GeometricMachineLearning
using JLD2

# the helper that reshapes the ensemble into consecutive `(q, p)` pairs
include("../utilities/parametric_data_helpers.jl")

# `smoke_size(full, smoke)` returns the second when GML_SMOKE is set. See scripts/README.md.
include("../utilities/smoke.jl")

# friction force coefficient
const nu = 0.001

# number of initial conditions per dimension, so `ni_dim^2` trajectories in total
const ni_dim = 2

# the time grid
const T = 13
const nt = smoke_size(100, 20)
const dt = T / nt

const width = 4
const nhidden = 3
const batch_size = smoke_size(5000, 8)
const n_epochs = smoke_size(100000, 2)

# next to the script, unless GML_OUTPUT_DIR says otherwise -- an absolute path from whoever ran it
# last is no use to anybody else
const path_out = joinpath(get(ENV, "GML_OUTPUT_DIR", pwd()),
    "damped_oscillator_network.jld2")

const IC = vec([(q = q0, p = p0) for q0 in range(-1, 1, ni_dim), p0 in range(-1, 1, ni_dim)])
const ni = ni_dim^2
const t = collect(dt * range(0, nt, step = 1))

# The analytic solution of the damped oscillator, one row per initial condition.
const omega = sqrt(4 - nu^2) / 2
const q = zeros(Float64, ni, nt + 1)
const p = zeros(Float64, ni, nt + 1)
for i in 1:(nt + 1)
    for j in 1:ni
        q[j, i] = (1 / omega) * (IC[j].p + nu / 2 * IC[j].q) * exp(-nu * t[i] / 2) *
                  sin(omega * t[i]) + IC[j].q * exp(-nu * t[i] / 2) * cos(omega * t[i])
        p[j, i] = -(1 / omega) * (IC[j].q + nu / 2 * IC[j].p) * exp(-nu * t[i] / 2) *
                  sin(omega * t[i]) + IC[j].p * exp(-nu * t[i] / 2) * cos(omega * t[i])
    end
end

const dl = DataLoader(turn_q_p_data_into_correct_format((q = q, p = p)))

const arch = ForcedSympNet(
    2; upscaling_dimension = width, n_layers = nhidden, forcing_type = :P)
const nn = NeuralNetwork(arch)

const loss_array = Optimizer(AdamOptimizer(), nn)(nn, dl, Batch(batch_size), n_epochs)

JLD2.save(path_out, "parameters", GeometricMachineLearning.map_to_cpu(nn.params),
    "training loss", loss_array, "ni_dim", ni_dim, "T", T, "nt", nt,
    "n_epochs", n_epochs, "width", width, "nhidden", nhidden,
    "batch_size", batch_size, "nu", nu)
