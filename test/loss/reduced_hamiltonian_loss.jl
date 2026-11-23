using GeometricMachineLearning
using GeometricMachineLearning: _reduced_hamiltonian_gradient, params
using Zygote: Zygote
using Test
using LinearAlgebra: norm, dot
import Random

include("../helpers/eltypes.jl")
include("../helpers/leaves.jl")

Random.seed!(1234)

# |p|²/2 + cos(q₁) + q₂²/2 on ℝ⁴, non-linear in every coordinate.
H(X) = vec(sum(abs2, X[3:4, :]; dims = 1)) ./ 2 .+ cos.(X[1, :]) .+ X[2, :] .^ 2 ./ 2

function make_decoder(::Type{T}) where {T}
    decoder(NeuralNetwork(
        SymplecticAutoencoder(4, 2; n_encoder_blocks = 2, n_decoder_blocks = 2), T))
end

# The reference gradient of H∘dec, by Zygote, one latent point at a time.
function reduced_gradient(dec, z)
    reduce(hcat,
        [Zygote.gradient(ζ -> H(reshape(dec(ζ), 4, 1))[1], collect(ζ))[1]
         for ζ in eachcol(z)])
end
targets(dec, z) = vcat(dec(z), reduced_gradient(dec, z))

# The central differences with the default step cbrt(eps(T)) have a truncation and a round-off
# error of order eps(T)^(2/3) each, relative to the size of the gradient.
fd_rtol(::Type{T}) where {T} = 20 * cbrt(eps(T))^2

relerr(a, b) = norm(a - b) / norm(b)

# The central differences agree with automatic differentiation, at the default step.
function test_gradient(::Type{T}) where {T}
    dec = make_decoder(T)
    z = randn(T, 2, 30)
    G = @inferred _reduced_hamiltonian_gradient(
        ReducedHamiltonianLoss(H), dec.model, params(dec), z)
    @test eltype(G) === T
    @test relerr(G, reduced_gradient(dec, z)) < fd_rtol(T)
end

# A given step reaches the central differences: their error is of order h², so halving a large
# step divides it by four.
function test_step(::Type{T}) where {T}
    dec = make_decoder(T)
    z = randn(T, 2, 30)
    G_ref = reduced_gradient(dec, z)
    err(h) = relerr(
        _reduced_hamiltonian_gradient(ReducedHamiltonianLoss(H; h = h), dec.model,
            params(dec), z),
        G_ref)
    @test err(0.2) / err(0.1) ≈ 4 rtol = 0.1
end

# A decoder on its own targets has a loss of the size of the difference error, of the element
# type of the data.
function test_self_consistency(::Type{T}) where {T}
    dec = make_decoder(T)
    z = randn(T, 2, 30)
    l = ReducedHamiltonianLoss(H)(dec, z, targets(dec, z))
    @test l isa T
    @test 0 <= l < fd_rtol(T)
end

# With λ = 0 the loss is the relative error of the states. λ weighs the gradient term only, and the
# gradient term vanishes on exact target gradients and is positive on perturbed ones.
function test_terms(::Type{T}) where {T}
    dec = make_decoder(T)
    z = randn(T, 2, 30)
    x, g = dec(z), reduced_gradient(dec, z)
    x_target = x .+ T(0.1) .* randn(T, size(x))
    g_target = g .+ T(0.1) .* randn(T, size(g))
    state_error = norm(x - x_target) / norm(x_target)

    @test ReducedHamiltonianLoss(H; λ = 0)(dec, z, vcat(x_target, g_target)) ≈ state_error rtol = 8 *
                                                                                                  eps(T)
    @test ReducedHamiltonianLoss(H)(dec, z, vcat(x_target, g)) ≈ state_error atol = fd_rtol(T)

    l0 = ReducedHamiltonianLoss(H; λ = 0)(dec, z, vcat(x_target, g_target))
    l1 = ReducedHamiltonianLoss(H; λ = 1)(dec, z, vcat(x_target, g_target))
    @test l1 - l0 > 10 * fd_rtol(T)
    for λ in (0.7, 3.0)
        lλ = ReducedHamiltonianLoss(H; λ = λ)(dec, z, vcat(x_target, g_target))
        @test lλ ≈ l0 + T(λ) * (l1 - l0) rtol = 8 * eps(T)
    end
end

# Where every target gradient of the batch vanishes, the loss is finite.
function test_zero_target_gradient(::Type{T}) where {T}
    dec = make_decoder(T)
    z = randn(T, 2, 10)
    l = ReducedHamiltonianLoss(H)(dec, z, vcat(dec(z), zeros(T, 2, 10)))
    @test l isa T
    @test isfinite(l)
end

# The 3-tensors of a `DataLoader` give the loss of the matrices.
function test_tensor_method(::Type{T}) where {T}
    dec = make_decoder(T)
    z = randn(T, 2, 30)
    out = targets(dec, z) .+ T(0.1) .* randn(T, 6, 30)
    loss = ReducedHamiltonianLoss(H; λ = 0.7)
    @test loss(dec.model, params(dec), reshape(z, 2, 30, 1), reshape(out, 6, 30, 1)) ≈
          loss(dec.model, params(dec), z, out) rtol = 8 * eps(T)
end

# The Zygote gradient with respect to the parameters agrees with a central difference of the loss
# in a random direction of the array leaves.
function test_parameter_gradient()
    dec = make_decoder(Float64)
    teacher = make_decoder(Float64)
    z = randn(2, 30)
    out = targets(teacher, z)
    loss = ReducedHamiltonianLoss(H; λ = 0.7)
    ps = params(dec)
    f(p) = loss(dec.model, p, z, out)
    dp = Zygote.gradient(f, ps)[1]
    leaves = Tuple{Array{Float64}, Any, Array{Float64}}[]
    foreach_leaf(ps, dp) do p, g
        p isa Array{Float64} && push!(leaves, (p, g, randn(size(p))))
    end
    @test !isempty(leaves)
    @test all(((_, g, _),) -> g !== nothing, leaves)
    ad = sum(dot(collect(g), d) for (_, g, d) in leaves)
    ε = 1e-5
    shift!(s) = foreach(((p, _, d),) -> p .+= s .* d, leaves)
    shift!(ε)
    fp = f(ps)
    shift!(-2ε)
    fm = f(ps)
    shift!(ε)
    fd = (fp - fm) / (2ε)
    # the central difference in ε has a truncation error of order ε², and the round-off of the
    # loss's own central differences in z, of order eps / h with h = cbrt(eps), divided by ε
    @test ad ≈ fd rtol = 10 * (ε^2 + eps() / (cbrt(eps()) * ε))
end

# One `train!` step on a `DataLoader` of the latent points and the targets gives a finite loss of
# the network's element type and moves the parameters.
function test_one_training_step(::Type{T}) where {T}
    dec = make_decoder(T)
    teacher = make_decoder(T)
    z = randn(T, 2, 40)
    dl = DataLoader(z, targets(teacher, z); suppress_info = true)
    before = deepcopy(dec.params)
    o = TrainingOptimizer(dec.params; algorithm = Adam())
    history = train!(dec, o, dl, Batch(40), 1; loss = ReducedHamiltonianLoss(H),
        show_progress = false)
    @test eltype(history) === T
    @test all(isfinite, history)
    moved = Bool[]
    foreach_leaf((p, q) -> push!(moved, p != q), dec.params, before)
    @test all(moved)
end

@testset "ReducedHamiltonianLoss rejects invalid weights, steps and outputs" begin
    re_λ(s) = Regex("^ArgumentError: ReducedHamiltonianLoss requires a finite weight λ >= 0, got λ = $s\\.\$")
    @test_throws re_λ("-1") ReducedHamiltonianLoss(H; λ = -1)
    @test_throws re_λ("NaN") ReducedHamiltonianLoss(H; λ = NaN)
    @test_throws re_λ("Inf") ReducedHamiltonianLoss(H; λ = Inf)
    re_h(s) = Regex("^ArgumentError: ReducedHamiltonianLoss requires a finite step h > 0, got h = $s\\.\$")
    @test_throws re_h("0") ReducedHamiltonianLoss(H; h = 0)
    @test_throws re_h("-0.0001") ReducedHamiltonianLoss(H; h = -1e-4)
    @test_throws re_h("NaN") ReducedHamiltonianLoss(H; h = NaN)
    @test_throws re_h("Inf") ReducedHamiltonianLoss(H; h = Inf)
    # the guards cannot be bypassed by a positional constructor
    @test_throws r"^MethodError: no method matching (GeometricMachineLearning\.)?ReducedHamiltonianLoss\(::typeof\(\S*H\), ::Float64, ::Float64\)\n" ReducedHamiltonianLoss(
        H, -1.0, 0.0)
    @test ReducedHamiltonianLoss(H; λ = 0).λ == 0

    dec = make_decoder(Float64)
    z = randn(2, 5)
    out = targets(dec, z)
    re_rows(r) = Regex("^ArgumentError: ReducedHamiltonianLoss requires an output of 6 rows, the 4 rows of the target states x\\* over the 2 rows of the target gradients g\\*, got $r rows\\.\$")
    @test_throws re_rows(4) ReducedHamiltonianLoss(H)(dec, z, out[1:4, :])
    @test_throws re_rows(8) ReducedHamiltonianLoss(H)(dec, z, vcat(out, zeros(2, 5)))
end

@testset "ReducedHamiltonianLoss in $T" for T in REAL_ELTYPES
    test_gradient(T)
    test_step(T)
    test_self_consistency(T)
    test_terms(T)
    test_zero_target_gradient(T)
    test_tensor_method(T)
    test_one_training_step(T)
end

@testset "ReducedHamiltonianLoss parameter gradient" begin
    test_parameter_gradient()
end
