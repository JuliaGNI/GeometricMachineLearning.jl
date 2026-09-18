# Train a `ParametricResNet` on the unforced harmonic oscillator, then integrate one trajectory with
# the learned network and plot it against the analytic solution.
#
# This is the baseline `GeneralizedHamiltonianArchitecture` is compared against: the same information
# reaches the network -- the state and the parameters of the system -- and nothing in it makes the
# map symplectic. Run it beside `forced_generalized_hamiltonian_neural_network.jl`.
using GeometricMachineLearning
using GeometricMachineLearning: ParametricLoss
using CairoMakie

# the helpers that reshape the ensemble into the layout `ParametricDataLoader` wants
include("../utilities/parametric_data_helpers.jl")

# `smoke_size(full, smoke)` returns the second when GML_SMOKE is set. See scripts/README.md.
include("../utilities/smoke.jl")

# natural frequency of the harmonic oscillator
const omega = 1.0

# The external forcing is switched off here, so `Omega` only has to be something other than `omega`:
# the analytic solution divides by `omega^2 - Omega^2`.
const Omega = 3.5
const F = 0.0

# number of initial conditions per dimension, so `ni_dim^2` trajectories in total
const ni_dim = smoke_size(10, 2)

# the time grid
const T = 2π * 5
const nt = smoke_size(1000, 20)
const dt = T / nt

const IC = vec([(q = q0, p = p0) for q0 in range(-1, 1, ni_dim), p0 in range(-1, 1, ni_dim)])
const t = collect(dt * range(0, nt, step = 1))

const q, p = forced_harmonic_oscillator_solution(t, IC; omega = omega, Omega = Omega, F = F)

const dl = load_time_dependent_harmonic_oscillator_with_parametric_data_loader(
    (q = q, p = p), t, IC)

const width = 2
const n_blocks = 1

# One `NamedTuple` of system parameters is enough: the architecture uses it for its shape. Passing
# `parameters` is what makes `ResNet` dispatch to `ParametricResNet`.
const system_parameters = turn_parameters_into_correct_format(t, IC)[1]
const arch = ResNet(2; n_blocks = n_blocks, width = width,
    activation = tanh, parameters = system_parameters)
const nn = NeuralNetwork(arch)

const batch_size = smoke_size(128, 8)
const n_epochs = smoke_size(200, 2)
const loss_array = Optimizer(AdamOptimizer(), nn)(
    nn, dl, Batch(batch_size), n_epochs, ParametricLoss())

# Integrate one trajectory and compare it against the analytic solution. The network is a map over
# one step, so the system parameters -- here the time -- are passed at each step with the state.
const trajectory_number = 1
const n_steps = nt
const trajectory = (q = zeros(1, n_steps), p = zeros(1, n_steps))
trajectory.q[:, 1] .= q[trajectory_number, 1]
trajectory.p[:, 1] .= p[trajectory_number, 1]
for t_step in 0:(n_steps - 2)
    qp_temporary = nn(
        (q = [trajectory.q[1, t_step + 1]], p = [trajectory.p[1, t_step + 1]]),
        (t = t[t_step + 1],))
    trajectory.q[:, t_step + 2] .= qp_temporary.q
    trajectory.p[:, t_step + 2] .= qp_temporary.p
end

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "time step", ylabel = "q")
lines!(ax, trajectory.q[1, :]; label = "network")
lines!(ax, q[trajectory_number, 1:n_steps]; label = "analytic")
axislegend(ax)
CairoMakie.save("parametric_resnet.png", fig)
