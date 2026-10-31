using Test
using GeometricMachineLearning: NetworkParameters, StiefelManifold, params

include("../helpers/eltypes.jl")

# The three-stage pipeline of `scripts/reproduction/multimodal_symplectic_autoencoder.jl`, at the
# sizes of its smoke mode. The file defines the stages and runs them only when it is the program,
# so the include below defines them here. Each mutant testset replaces one part of a stage by a
# defect and asserts that the script's own assertion for that stage fails, with its whole message.
withenv("GML_SMOKE" => "1") do
    include(joinpath(@__DIR__, "..", "..", "scripts", "reproduction",
        "multimodal_symplectic_autoencoder.jl"))
end

# Euclidean Adam on the parent arrays of every leaf, `StiefelManifold` leaves included: the step
# that leaves the manifold.
function euclidean_adam_step(nn; β₁ = 0.9, β₂ = 0.99, δ = 1e-8)
    T = parameter_eltype(params(nn))
    moments = IdDict{Any, Tuple{Any, Any}}()
    k = Ref(0)
    function step!(x::AbstractArray, g::AbstractArray)
        a = x isa StiefelManifold ? parent(x) : x
        m, v = get!(() -> (zero(a), zero(a)), moments, a)
        @. m = T(β₁) * m + (1 - T(β₁)) * g
        @. v = T(β₂) * v + (1 - T(β₂)) * g^2
        @. a -= T(STEP_SIZE) * (m / (1 - T(β₁)^k[])) / (sqrt(v / (1 - T(β₂)^k[])) + T(δ))
        nothing
    end
    step!(x::NamedTuple, g::NamedTuple) = foreach(step!, values(x), values(g))
    function (ps::NetworkParameters, dp::NetworkParameters)
        k[] += 1
        step!(params(ps), params(dp))
    end
end

# The Stage II loss with a cycle term in place of `CrossModalLoss` and no particle term: the
# reconstruction term plus `β_cycle ‖encode(grid, decode(grid, z)) - z‖²`, with `z` the latent
# state of the DLRA snapshots and `β_cycle = 0.1`. The core is weight-tied, so the cycle term is
# round-off, and Stage I leaves the reconstruction term at its SVD truncation error: the sum
# stays near its value after Stage I. Without the reconstruction term, the assertion would compare
# two round-off values.
function cycle_loss(data, particles)
    T = eltype(data.δF)
    reconstruction = CharbonnierRelativeLoss()
    function loss(model, ps)
        z = encode(model, ps, :dlra, data.dlra)
        reconstruction(model, ps, :grid, data.δF) +
        T(0.1) * sum(abs2, encode(model, ps, :grid, decode(model, ps, :grid, z)) .- z)
    end
    (; loss, extra = nothing)
end

# The cross-reconstruction term alone: `β_align = 0`, no reconstruction term and no further term.
function cross_reconstruction_loss(data, particles)
    cross = CrossModalLoss(:dlra, :grid; β_align = 0.0)
    (; loss = (model, ps) -> cross(model, ps, data.dlra, data.δF), extra = nothing)
end

# The loss that Stage II asserts on, computed from the library's losses: the
# `CharbonnierRelativeLoss` of the grid modality plus `CrossModalLoss(:dlra, :grid)`.
function clause_loss(nn, data)
    reconstruction = CharbonnierRelativeLoss()(nn.model, params(nn), :grid, data.δF)
    reconstruction + CrossModalLoss(:dlra, :grid)(nn.model, params(nn), data.dlra, data.δF)
end

@testset "The three stages meet their assertions ($T)" for T in REAL_ELTYPES
    result = run_pipeline(T; verbose = false)
    @test result.ε_I isa T
    @test result.ε_I < PROJECTION_TOLERANCE
    @test eltype(result.losses) === T
    @test length(result.losses) == N_STAGE_II_STEPS + 1
    @test result.losses[end] <= LOSS_REDUCTION * result.losses[1]
    # The losses are reconstruction plus `CrossModalLoss`, without the particle term, after Stage I
    # and after the last step.
    data = synthetic_dataset(T)
    @test isapprox(
        result.losses[1], clause_loss(stage_I!(multimodal_network(T), data), data);
        rtol = 8 * eps(T))
    @test isapprox(result.losses[end], clause_loss(result.nn, data); rtol = 8 * eps(T))
    @test result.defect <= ORTHONORMALITY_EPS * eps(T)
    # The implicit midpoint rule has order 2: halving the step divides the energy error by 4.
    @test eltype(result.energy_errors) === T
    @test ENERGY_RATIO_RANGE[1] <= result.ratio <= ENERGY_RATIO_RANGE[2]
    @test result.decoded_error <= result.ε_I
end

# One leaf off the manifold is enough: scaling the two columns of `Φx` by `1 + δ` gives
# `‖YᵀY - I‖_F = ((1 + δ)² - 1) √2`, and every other leaf stays orthonormal.
@testset "The orthonormality defect is that of the worst leaf ($T)" for T in REAL_ELTYPES
    nn = multimodal_network(T)
    @test orthonormality_defect(nn) <= ORTHONORMALITY_EPS * eps(T)
    δ = T(1e-3)
    parent(params(nn).frontends.grid.Φx) .*= 1 + δ
    @test orthonormality_defect(nn) isa T
    @test isapprox(orthonormality_defect(nn), ((1 + δ)^2 - 1) * sqrt(T(2)); rtol = 1e-3)
end

@testset "M1: without Stage I the projection error is not below the tolerance ($T)" for T in REAL_ELTYPES
    @test_throws r"^AssertionError: Stage I: the projection error \S+ is not below 0\.01\.$" run_pipeline(
        T; verbose = false, initialize! = (nn, data) -> nn)
end

@testset "M2: Euclidean Adam takes the Stiefel leaves off the manifold ($T)" for T in REAL_ELTYPES
    @test_throws Regex("^AssertionError: Stage II: a Stiefel leaf has ‖YᵀY - I‖ = \\S+, more " *
                       "than 50 eps\\($T\\)\\.\$") run_pipeline(
        T; verbose = false, stage_II_step = euclidean_adam_step)
end

@testset "M3: without the reference state the background is not decoded ($T)" for T in REAL_ELTYPES
    @test_throws r"^AssertionError: Stage III: the decoded initial field has relative error \S+, more than the Stage I projection error \S+\.$" run_pipeline(
        T; verbose = false, shift = false)
end

@testset "M4: the cycle term does not move the loss ($T)" for T in REAL_ELTYPES
    @test_throws r"^AssertionError: Stage II: the loss went from \S+ to \S+, more than 0\.5 of its value after Stage I\.$" run_pipeline(
        T; verbose = false, stage_II_loss = cycle_loss)
end

@testset "M4 contrast: the cross-reconstruction term halves the loss ($T)" for T in REAL_ELTYPES
    result = run_pipeline(T; verbose = false, stage_II_loss = cross_reconstruction_loss)
    @test result.losses[end] <= LOSS_REDUCTION * result.losses[1]
end
