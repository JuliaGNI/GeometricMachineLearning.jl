# The three-stage training curriculum of a multi-modal symplectic autoencoder, on a synthetic
# coupled particle + grid + DLRA 1D1V dataset, in `Float32` and `Float64`.
#
#   Stage I    `solve!` sets the grid frontend's Stiefel bases from the SVD of the fluctuation
#              snapshots and the core's symplectic lift from the cotangent lift of their frontend
#              coordinates; the DLRA frontend takes the grid frontend's bases.
#   Stage II   joint training: the reconstruction loss of the grid modality,
#              `CrossModalLoss(:dlra, :grid)` and the error of the particle states translated from
#              the grid modality, stepped with
#              `CompositeMethod(; manifold = FactoredStiefelAdam(), array = Adam())`.
#   Stage III  the reduced system `HRedSys` of a pendulum Hamiltonian on the frontend
#              coordinates, shifted by the background, integrated with `ImplicitMidpoint`.
#
# The dataset is synthetic. It tests that the pipeline runs and that each stage does what it
# claims; it says nothing about the physics of a plasma.
#
# Usage, from the repository root:
#
#   julia --startup-file=no --project=scripts scripts/reproduction/multimodal_symplectic_autoencoder.jl
#
# With `GML_SMOKE` set, the script takes the smaller sizes of `smoke_size` and runs every
# assertion. `test/integration/multimodal_pipeline.jl` includes this file and calls its functions;
# the stages run only when the file is the program.

using GeometricMachineLearning
using GeometricMachineLearning: params
using GeometricOptimizers: solve!, CompositeMethod, FactoredStiefelAdam, Adam,
                           TrainingOptimizer, optimization_step!
using GeometricIntegrators: ImplicitMidpoint, StateVariable
using LinearAlgebra: I, norm, qr, svd, Diagonal, diag, dot
using Random: Xoshiro
using Zygote: Zygote

# The time limit counts from here, the end of loading, to the end of Stage III in both
# precisions, compilation included.
const LOADED_AT = time()

include("../utilities/smoke.jl")

# The grid is `NX × NV`, and both frontends keep `MX × MV` modes per parity, so the frontend
# coordinates have dimension `2N = 2 MX MV`; the core reduces them to `2n = N_LATENT`.
const NX, NV = 16, 16
const MX, MV = 2, 2
const N_FRONTEND = 2 * MX * MV
const N_LATENT = 4
# The particle frontend maps `N_PARTICLES` particles in one dimension to `N_FRONTEND ÷ 2` slots.
const N_PARTICLES = 8
const N_SNAPSHOTS = smoke_size(32, 8)
const N_STAGE_II_STEPS = smoke_size(1000, 400)
const STEP_SIZE = 1e-3
const PARTICLE_DISTANCE = 0.3
const NOISE = 1e-3
# Stage III integrates over `(0, TIME_INTERVAL)` in each of `N_STAGE_III_STEPS` steps.
const TIME_INTERVAL = 2.0
const N_STAGE_III_STEPS = (20, 40)
const SEED = 42

# The bounds that the stages assert.
const PROJECTION_TOLERANCE = 1e-2
const LOSS_REDUCTION = 0.5
const ORTHONORMALITY_EPS = 50
const ENERGY_RATIO_RANGE = (3, 5)
const TIME_LIMIT = smoke_size(180, 60)

normalized(a) = a ./ norm(a)

# The full field of an even part `Fp` and an odd part `Fm` on the positive half of the velocity
# grid: `F[:, j] = (Fp + Fm) / √2` and `F[:, NV + 1 - j] = (Fp - Fm) / √2` for `j ≤ NV / 2`.
function recombine(Fp::AbstractMatrix{T}, Fm::AbstractMatrix{T}) where {T}
    h = NV ÷ 2
    F = similar(Fp, NX, NV)
    F[:, 1:h] = (Fp + Fm) / sqrt(T(2))
    F[:, NV:-1:(h + 1)] = (Fp - Fm) / sqrt(T(2))
    F
end

"""
    synthetic_dataset(T; n_snapshots = N_SNAPSHOTS)

Return the fields of the snapshots, `F = f0 + δF`, with `f0` a background even in `v` and `δF`
a fluctuation of two spatial modes, each with an even and an odd velocity profile, plus noise.
`dlra` holds the rank-4 factors `(X, S, V)` of each `δF`, batched along axis 3, and
`particle_perturbation` the matrix that sets the relative basis of the particle states (see
`particle_snapshots`).
"""
function synthetic_dataset(::Type{T}; n_snapshots::Integer = N_SNAPSHOTS) where {T}
    rng = Xoshiro(SEED)
    h = NV ÷ 2
    x = T.(range(0, 2π; length = NX + 1)[1:NX])
    v = T.(range(-4, 4; length = NV))[1:h]
    X1, X2 = normalized(cos.(x)), normalized(sin.(x))
    ψ1 = normalized(exp.(-v .^ 2 ./ 4))
    ψ2 = normalized(v .^ 2 .* exp.(-v .^ 2 ./ 4))
    ψ2 = normalized(ψ2 - dot(ψ1, ψ2) * ψ1)
    f0 = recombine(sqrt(T(2)) .* (1 .+ sin.(x) ./ 2) * transpose(exp.(-v .^ 2 ./ 2)),
        zeros(T, NX, h))
    δF = zeros(T, NX, NV, n_snapshots)
    for (b, t) in enumerate(T.(range(0, 2π; length = n_snapshots)))
        a1, b1 = cos(t), -sin(t)
        a2, b2 = T(0.6) * cos(2t + T(0.3)), -T(0.6) * sin(2t + T(0.3))
        Fp = a1 * X1 * transpose(ψ1) + a2 * X2 * transpose(ψ2)
        Fm = b1 * X1 * transpose(ψ1) + b2 * X2 * transpose(ψ2)
        δF[:, :, b] = recombine(Fp, Fm) + T(NOISE) * randn(rng, T, NX, NV)
    end
    r = MX * MV
    factors = map(1:n_snapshots) do b
        s = svd(δF[:, :, b])
        (s.U[:, 1:r], Matrix(Diagonal(s.S[1:r])), s.V[:, 1:r])
    end
    dlra = (X = stack(first.(factors)), S = stack(getindex.(factors, 2)),
        V = stack(last.(factors)))
    particle_perturbation = randn(rng, T, N_PARTICLES - 1, N_FRONTEND ÷ 2 - 1)
    (; f0, δF, F = f0 .+ δF, dlra, particle_perturbation)
end

"""
    multimodal_network(T)

Return the network: the frontends `:grid` (`BilinearStiefelLayer`), `:dlra`
(`LowRankStiefelLayer`) and `:particles` (`CanonicalSlotLayer`) on a
`WeightTiedSymplecticAutoencoder` core from `N_FRONTEND` to `N_LATENT`.
"""
function multimodal_network(::Type{T}) where {T}
    frontends = (grid = BilinearStiefelLayer(NX, NV, MX, MV),
        dlra = LowRankStiefelLayer(NX, NV, MX, MV; r_out = MX * MV),
        particles = CanonicalSlotLayer(N_PARTICLES, 1, N_FRONTEND ÷ 2))
    core = WeightTiedSymplecticAutoencoder(N_FRONTEND, N_LATENT; n_blocks = 1,
        n_layers = 2, n_ambient_layers = 0, sympnet_upscale = 4)
    NeuralNetwork(MultiModalSymplecticAutoencoder(frontends, core), CPU(), T;
        rng = Xoshiro(SEED))
end

# The frontend coordinates of a field `F` in the grid modality, as one vector `(q, p)`.
function grid_coordinates(nn, F::AbstractMatrix)
    z = nn.architecture.frontends.grid(F, params(nn).frontends.grid)
    vcat(z.q, z.p)
end

# The field of the frontend coordinates `z`, through the inverse of the grid frontend.
function grid_field(nn, z::AbstractVector)
    N = length(z) ÷ 2
    inverse_layer(nn.architecture.frontends.grid)((q = z[1:N], p = z[(N + 1):end]),
        params(nn).frontends.grid)
end

# The frontend coordinates of every fluctuation snapshot, one column per snapshot.
function grid_coordinates(nn, data::NamedTuple)
    stack(b -> grid_coordinates(nn, data.δF[:, :, b]), axes(data.δF, 3))
end

"""
    stage_I!(nn, data)

Set the linear parts of the network from SVDs with `solve!`: the Stiefel bases of the grid
frontend from the fluctuations `data.δF`, and the core's symplectic lift from the cotangent lift
of the snapshots' frontend coordinates. The DLRA frontend takes the grid frontend's bases.

The core is weight-tied, so its decode–encode is the projection of its lift whatever the
gradient layers between; after this stage the network reconstructs the fluctuations to their
SVD truncation error.
"""
function stage_I!(nn, data)
    T = eltype(data.δF)
    ps = params(nn)
    solve!(nn.architecture.frontends.grid, ps.frontends.grid, data.δF)
    copyto!(parent(ps.frontends.dlra.Φx), parent(ps.frontends.grid.Φx))
    copyto!(parent(ps.frontends.dlra.Φv), parent(ps.frontends.grid.Φv))
    psd = NeuralNetwork(PSDArch(N_FRONTEND, N_LATENT), CPU(), T)
    solve!(psd, grid_coordinates(nn, data))
    copyto!(parent(ps.core[end].weight), parent(params(psd)[1].weight))
    nn
end

"""
    projection_error(nn, data)

Return `max_b ‖F_b - (f0 + Π(F_b - f0))‖ / ‖F_b‖`, with `Π` the decode–encode of the grid
modality through the whole network.
"""
function projection_error(nn, data)
    maximum(axes(data.δF, 3)) do b
        δF = data.δF[:, :, b]
        norm(δF - decode(nn, :grid, encode(nn, :grid, δF))) / norm(data.F[:, :, b])
    end
end

"""
    particle_snapshots(nn, data)

Return the particle states, `(q, p)` of size `(1, N_PARTICLES, B)`, of the snapshots: the
inverse of the particle frontend applied to the grid snapshots' frontend coordinates. The three
modalities of a snapshot then describe one latent state.

The relative basis of these states is not the network's: it is the network's basis plus
`PARTICLE_DISTANCE` times `data.particle_perturbation`, orthonormalized, and Stage II learns it.
Adam moves each parameter by about the step size per step, so the distance is one that the steps
of Stage II cover.
"""
function particle_snapshots(nn, data)
    T = eltype(data.δF)
    Φ_rel = parent(params(nn).frontends.particles.Φ_rel)
    factors = qr(Φ_rel + T(PARTICLE_DISTANCE) * data.particle_perturbation)
    basis = Matrix(factors.Q) * Diagonal(sign.(diag(factors.R)))
    inv_slots = inverse_layer(nn.architecture.frontends.particles)
    x = inv_slots(grid_coordinates(nn, data), (Φ_rel = basis,))
    Nq = size(x, 1) ÷ 2
    (q = reshape(x[1:Nq, :], 1, N_PARTICLES, :),
        p = reshape(x[(Nq + 1):end, :], 1, N_PARTICLES, :))
end

# A particle state `(q, p)` of size `(1, N_PARTICLES, B)` as the matrix that `decode` returns.
particle_matrix(x) = vcat(reshape(x.q, N_PARTICLES, :), reshape(x.p, N_PARTICLES, :))

"""
    joint_loss(data, particles)

Return the Stage II loss `(model, ps) -> L` over the whole batch: the `CharbonnierRelativeLoss`
of the grid modality, `CrossModalLoss(:dlra, :grid)`, and the relative squared error of the
particle states translated from the grid modality.

The particle term translates and does not encode the particle states: the particle frontend sorts
its input, and a decoded particle state need not be in the order that the sort keeps.
"""
function joint_loss(data, particles)
    reconstruction = CharbonnierRelativeLoss()
    dlra_to_grid = CrossModalLoss(:dlra, :grid)
    target = particle_matrix(particles)
    target_norm² = sum(abs2, target)
    function loss(model, ps)
        reconstruction(model, ps, :grid, data.δF) +
        dlra_to_grid(model, ps, data.dlra, data.δF) +
        sum(abs2, translate(model, ps, :grid, :particles, data.δF) .- target) / target_norm²
    end
end

parameter_eltype(::NetworkParameters{T}) where {T} = T

# The default step of Stage II: the manifold leaves under `FactoredStiefelAdam`, the others under
# `Adam`, both with the step size `STEP_SIZE`.
function composite_step(nn)
    T = parameter_eltype(params(nn))
    optimizer = TrainingOptimizer(params(nn);
        algorithm = CompositeMethod(; manifold = FactoredStiefelAdam(), array = Adam()),
        linesearch = T(STEP_SIZE))
    (ps, dp) -> optimization_step!(ps, optimizer, dp)
end

"""
    stage_II!(nn, loss; n_steps = N_STAGE_II_STEPS, step! = composite_step(nn), verbose = false)

Take `n_steps` steps of `step!(params(nn), gradient)` on `loss(nn.model, params(nn))`, and
return the loss before the first step and after each step.
"""
function stage_II!(nn, loss; n_steps::Integer = N_STAGE_II_STEPS,
        step! = composite_step(nn), verbose::Bool = false)
    ps = params(nn)
    losses = [loss(nn.model, ps)]
    for k in 1:n_steps
        dp = Zygote.gradient(p -> loss(nn.model, p), ps)[1]
        step!(ps, dp)
        push!(losses, loss(nn.model, ps))
        verbose && println("  step ", k, ": loss ", losses[end])
    end
    losses
end

stiefel_leaves(Y::StiefelManifold) = [Y]
function stiefel_leaves(t::Union{NamedTuple, Tuple})
    reduce(vcat, map(stiefel_leaves, values(t));
        init = StiefelManifold[])
end
stiefel_leaves(::Any) = StiefelManifold[]

"""
    orthonormality_defect(nn)

Return the largest `‖YᵀY - I‖_F` over the `StiefelManifold` leaves of `params(nn)`.
"""
function orthonormality_defect(nn)
    leaves = stiefel_leaves(params(params(nn)))
    maximum(Y -> norm(transpose(parent(Y)) * parent(Y) - I), leaves)
end

# The pendulum Hamiltonian `H(q, p) = |p|²/2 + Σᵢ (1 - cos qᵢ)` on the frontend coordinates.
pendulum_v!(v, t, q, p, parameters) = (v .= p; nothing)
pendulum_f!(f, t, q, p, parameters) = (f .= .-sin.(q); nothing)
pendulum_h(t, q, p, parameters) = sum(abs2, p) / 2 + sum(1 .- cos.(q))

"""
    stage_III(nn, data; shift = true)

Integrate the reduced system of the pendulum Hamiltonian on the frontend coordinates over
`(0, TIME_INTERVAL)`, from the first snapshot, once in each of `N_STAGE_III_STEPS` steps.

With `shift = true`, the reference state `q_ref, p_ref` is the frontend coordinates of the
background `f0`, and a field is decoded as `f0` plus the field of the decoded coordinates less
the reference. With `shift = false`, the reference state and the field added on decoding are
both zero.

Return the largest error of the reduced energy along each rollout, and the relative error of the
decoded initial field `‖decode(encode(F₀)) - F₀‖ / ‖F₀‖`.
"""
function stage_III(nn, data; shift::Bool = true)
    T = eltype(data.F)
    N = N_FRONTEND ÷ 2
    F0 = data.F[:, :, 1]
    z0 = grid_coordinates(nn, F0)
    z_ref = shift ? grid_coordinates(nn, data.f0) : zeros(T, 2N)
    f_ref = shift ? data.f0 : zero(data.f0)
    ics = (q = StateVariable(z0[1:N]), p = StateVariable(z0[(N + 1):end]))
    systems = map(N_STAGE_III_STEPS) do n_steps
        HRedSys(2N, N_LATENT, encoder(nn), decoder(nn), pendulum_v!, pendulum_f!,
            pendulum_h, (zero(T), T(TIME_INTERVAL)), T(TIME_INTERVAL) / n_steps, ics;
            integrator = ImplicitMidpoint(), q_ref = z_ref[1:N], p_ref = z_ref[(N + 1):end],
            pullback_mode = :vjp)
    end
    energy_errors = map(systems) do rs
        sol = integrate_reduced_system(rs)
        H = [rs.h_reduced(zero(T), sol.q[k], sol.p[k], rs.parameters)
             for k in 0:(length(sol.t) - 1)]
        maximum(h -> abs(h - H[1]), H)
    end
    ẑ0 = GeometricMachineLearning.decode(systems[1],
        GeometricMachineLearning.encode(systems[1], z0))
    F̂0 = f_ref + grid_field(nn, ẑ0 - z_ref)
    (; energy_errors, decoded_error = norm(F̂0 - F0) / norm(F0))
end

"""
    run_pipeline(T; verbose = true, initialize! = stage_I!, stage_II_loss = joint_loss,
        stage_II_step = composite_step, shift = true)

Run the three stages in precision `T`, assert each stage's claim, and return what they measured.

The keywords name the parts of the stages: `initialize!(nn, data)` is Stage I,
`stage_II_loss(data, particles)` returns the loss of Stage II and `stage_II_step(nn)` its step,
and `shift` is passed to `stage_III`. The test of this script replaces each of them by a
defect and asserts that the matching assertion fails.
"""
function run_pipeline(::Type{T}; verbose::Bool = true, initialize! = stage_I!,
        stage_II_loss = joint_loss, stage_II_step = composite_step,
        shift::Bool = true) where {T}
    started = time()
    data = synthetic_dataset(T)
    nn = multimodal_network(T)

    initialize!(nn, data)
    ε_I = projection_error(nn, data)
    verbose && println(T, " Stage I: projection error ", ε_I, " (", seconds(started), ")")
    @assert ε_I < PROJECTION_TOLERANCE "Stage I: the projection error $ε_I is not below $PROJECTION_TOLERANCE."

    started = time()
    particles = particle_snapshots(nn, data)
    losses = stage_II!(nn, stage_II_loss(data, particles);
        step! = stage_II_step(nn), verbose = verbose)
    defect = orthonormality_defect(nn)
    verbose && println(T, " Stage II: loss ", losses[1], " -> ", losses[end],
        ", ratio ", losses[end] / losses[1], ", orthonormality defect ", defect,
        " (", seconds(started), ")")
    @assert losses[end] <= LOSS_REDUCTION * losses[1] "Stage II: the loss went from $(losses[1]) to $(losses[end]), more than $LOSS_REDUCTION of its value after Stage I."
    @assert defect <= ORTHONORMALITY_EPS * eps(T) "Stage II: a Stiefel leaf has ‖YᵀY - I‖ = $defect, more than $ORTHONORMALITY_EPS eps($T)."

    started = time()
    stage = stage_III(nn, data; shift = shift)
    ratio = stage.energy_errors[1] / stage.energy_errors[2]
    verbose &&
        println(T, " Stage III: energy errors ", stage.energy_errors, ", ratio ", ratio,
            ", decoded initial error ", stage.decoded_error, " (bound ε_I = ", ε_I, ") (",
            seconds(started), ")")
    @assert ENERGY_RATIO_RANGE[1] <= ratio <= ENERGY_RATIO_RANGE[2] "Stage III: halving the time step divides the energy error by $ratio, outside $ENERGY_RATIO_RANGE."
    @assert stage.decoded_error <= ε_I "Stage III: the decoded initial field has relative error $(stage.decoded_error), more than the Stage I projection error $ε_I."

    (; ε_I, losses, defect, stage..., ratio)
end

seconds(started) = string(round(time() - started; digits = 1), " s")

function main()
    for T in (Float32, Float64)
        run_pipeline(T)
    end
    elapsed = time() - LOADED_AT
    println("time from the end of `using` to the end of Stage III: ", seconds(LOADED_AT))
    @assert elapsed <= TIME_LIMIT "The stages took $(round(elapsed; digits = 1)) s, more than $TIME_LIMIT s."
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
