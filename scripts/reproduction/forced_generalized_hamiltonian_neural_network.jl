# Train the three forcings of `ForcedGeneralizedHamiltonianArchitecture` on the sinusoidally forced
# harmonic oscillator, then integrate one trajectory with the learned network and plot it against
# the analytic solution.
#
# Time is the parameter of the system here: the forcing is explicitly time dependent, so the network
# takes `(q, p)` and `t`, and the `ParametricDataLoader` carries one `(t = …,)` per column.
using GeometricMachineLearning
using GeometricMachineLearning: ParametricLoss
using CairoMakie

# the helpers that reshape the ensemble into the layout `ParametricDataLoader` wants
include("../utilities/parametric_data_helpers.jl")

# `smoke_size(full, smoke)` returns the second when GML_SMOKE is set. See scripts/README.md.
include("../utilities/smoke.jl")

# natural frequency of the harmonic oscillator
const omega = 1.0

# frequency and amplitude of the external sinusoidal forcing
const Omega = 3.5
const F = 0.9

# number of initial conditions per dimension, so `ni_dim^2` trajectories in total
const ni_dim = smoke_size(10, 2)

# the time grid
const T = 2π * 20
const nt = smoke_size(1000, 20)
const dt = T / nt

const IC = vec([(q = q0, p = p0) for q0 in range(-1, 1, ni_dim), p0 in range(-1, 1, ni_dim)])
const t = collect(dt * range(0, nt, step = 1))

const q, p = forced_harmonic_oscillator_solution(t, IC; omega = omega, Omega = Omega, F = F)

const dl = load_time_dependent_harmonic_oscillator_with_parametric_data_loader(
    (q = q, p = p), t, IC)

# The three forcings. `Q`, `P` and `QP` say what the forcing *depends on*, not what it changes: all
# three add to `p` and leave `q` alone, which is what a force does to the `ṗ` equation.
const width = 1
const nhidden = 1
const n_integrators = 2

# One `NamedTuple` of system parameters is enough: the architecture uses it for its shape.
const system_parameters = turn_parameters_into_correct_format(t, IC)[1]

function forced_architecture(forcing_type, layer_width)
    ForcedGeneralizedHamiltonianArchitecture(
        2; activation = tanh, width = layer_width, nhidden = nhidden,
        n_integrators = n_integrators, parameters = system_parameters,
        forcing_type = forcing_type)
end

const networks = (P = NeuralNetwork(forced_architecture(:P, width)),
    Q = NeuralNetwork(forced_architecture(:Q, width)),
    QP = NeuralNetwork(forced_architecture(:QP, 2width)))

const batch_size = smoke_size(128, 8)
const n_epochs = smoke_size(200, 2)
const batch = Batch(batch_size)
const loss = ParametricLoss()

# The default `ZygotePullback`, and not a `SymbolicPullback`. `n_integrators = 2` puts the symbolic
# construction past 10⁹ terms, so `_check_symbolic_pullback_is_tractable` refuses it -- see issue
# #245 and the layerwise construction that lifts it.
const loss_arrays = map(
    nn -> Optimizer(AdamOptimizer(), nn)(nn, dl, batch, n_epochs, loss), networks)

# Integrate one trajectory with the `QP` network and compare it against the analytic solution. The
# network is a map over one step, so the system parameters -- here the time -- are passed at each
# step alongside the state.
const trajectory_number = 1
const n_steps = nt
const trajectory = (q = zeros(1, n_steps), p = zeros(1, n_steps))
trajectory.q[:, 1] .= q[trajectory_number, 1]
trajectory.p[:, 1] .= p[trajectory_number, 1]
for t_step in 0:(n_steps - 2)
    qp_temporary = networks.QP(
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
CairoMakie.save("forced_generalized_hamiltonian_neural_network.png", fig)
