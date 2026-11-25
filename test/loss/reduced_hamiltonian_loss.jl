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

# The loss is the relative error of the states plus λ times the root-mean-square error of the
# gradient over the batch, divided by gradient_scale; the reference takes the gradient by Zygote.
# The central differences add at most λ * fd_rtol * ‖G‖ / (√B * s) to the gradient term.
function test_value(::Type{T}) where {T}
    dec = make_decoder(T)
    B = 30
    z = randn(T, 2, B)
    x, G = dec(z), reduced_gradient(dec, z)
    x_target = x .+ T(0.1) .* randn(T, size(x))
    g_target = G .+ T(0.1) .* randn(T, size(G))
    λ, s = 0.7, 2.5
    l = ReducedHamiltonianLoss(H; λ = λ, gradient_scale = s)(dec, z, vcat(x_target, g_target))
    reference = norm(x - x_target) / norm(x_target) +
                T(λ) * norm(G - g_target) / (sqrt(T(B)) * T(s))
    @test l isa T
    @test l ≈ reference atol = T(λ) * fd_rtol(T) * norm(G) / (sqrt(T(B)) * T(s))
end

# The gradient term is continuous in g*, also at g* = 0: for g* = σ·1, the triangle inequality
# bounds the change from σ = 0 by λ σ √(2n) / s.
function test_continuity_at_zero_target(::Type{T}) where {T}
    dec = make_decoder(T)
    z = randn(T, 2, 10)
    λ, s = 0.7, 2.5
    loss = ReducedHamiltonianLoss(H; λ = λ, gradient_scale = s)
    l(σ) = loss(dec, z, vcat(dec(z), fill(T(σ), 2, 10)))
    l0 = l(0)
    @test l0 isa T
    @test isfinite(l0) && l0 > 0
    for σ in (1e-6, 1e-12)
        @test abs(l(σ) - l0) <= T(λ) * T(σ) * sqrt(T(2)) / T(s) + 8 * eps(T) * l0
    end
end

# The value does not depend on the size of the batch: the batch [z z] has the loss of z.
function test_duplicated_batch(::Type{T}) where {T}
    dec = make_decoder(T)
    z = randn(T, 2, 30)
    out = targets(dec, z) .+ T(0.1) .* randn(T, 6, 30)
    loss = ReducedHamiltonianLoss(H; λ = 0.7, gradient_scale = 2.5)
    @test loss(dec, hcat(z, z), hcat(out, out)) ≈ loss(dec, z, out) rtol = 8 * eps(T)
end

# gradient_scale divides the gradient term and nothing else: the loss is l0 + (λ / s) (l1 - l0)
# for the loss l0 at λ = 0 and l1 at λ = 1, s = 1.
function test_gradient_scale(::Type{T}) where {T}
    dec = make_decoder(T)
    z = randn(T, 2, 30)
    out = targets(dec, z) .+ T(0.1) .* randn(T, 6, 30)
    l0 = ReducedHamiltonianLoss(H; λ = 0, gradient_scale = 2.5)(dec, z, out)
    l1 = ReducedHamiltonianLoss(H)(dec, z, out)
    @test l1 - l0 > 10 * fd_rtol(T)
    for (λ, s) in Iterators.product((0.7, 3.0), (0.4, 2.5))
        lλs = ReducedHamiltonianLoss(H; λ = λ, gradient_scale = s)(dec, z, out)
        @test lλs ≈ l0 + T(λ) / T(s) * (l1 - l0) rtol = 8 * eps(T)
    end
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
    # the central difference in ε has a relative truncation error of order ε², and an absolute
    # error from the round-off of the loss's own central differences in z, of order eps / h with
    # h = cbrt(eps), divided by ε; the loss is of order one
    @test ad ≈ fd rtol=10*ε^2 atol=10*eps()/(cbrt(eps())*ε)
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
    re_s(s) = Regex("^ArgumentError: ReducedHamiltonianLoss requires a finite gradient_scale > 0, got gradient_scale = $s\\.\$")
    @test_throws re_s("0") ReducedHamiltonianLoss(H; gradient_scale = 0)
    @test_throws re_s("-1.0") ReducedHamiltonianLoss(H; gradient_scale = -1.0)
    @test_throws re_s("NaN") ReducedHamiltonianLoss(H; gradient_scale = NaN)
    @test_throws re_s("Inf") ReducedHamiltonianLoss(H; gradient_scale = Inf)
    @test ReducedHamiltonianLoss(H; gradient_scale = 2).gradient_scale === 2.0
    @test ReducedHamiltonianLoss(H).gradient_scale === 1.0
    # the guards cannot be bypassed by a positional constructor
    @test_throws r"^MethodError: no method matching (GeometricMachineLearning\.)?ReducedHamiltonianLoss\(::typeof\(\S*H\), ::Float64, ::Float64, ::Float64\)\n" ReducedHamiltonianLoss(
        H, -1.0, 0.0, 0.0)
    @test ReducedHamiltonianLoss(H; λ = 0).λ == 0

    dec = make_decoder(Float64)
    z = randn(2, 5)
    out = targets(dec, z)
    re_rows(r) = Regex("^ArgumentError: ReducedHamiltonianLoss requires an output of 6 rows, the 4 rows of the target states x\\* over the 2 rows of the target gradients g\\*, got $r rows\\.\$")
    @test_throws re_rows(4) ReducedHamiltonianLoss(H)(dec, z, out[1:4, :])
    @test_throws re_rows(8) ReducedHamiltonianLoss(H)(dec, z, vcat(out, zeros(2, 5)))

    # a step that is finite and positive in Float64 but not in the element type of the data
    dec32 = make_decoder(Float32)
    z32 = randn(Float32, 2, 5)
    out32 = targets(dec32, z32)
    function re_step(h, hT)
        Regex("^ArgumentError: The step h = $h is $hT in the element type Float32 of the data; it has to be finite and > 0 there\\.\$")
    end
    @test_throws re_step("1.0e-50", "0.0") ReducedHamiltonianLoss(H; h = 1e-50)(dec32, z32, out32)
    @test_throws re_step("1.0e300", "Inf") ReducedHamiltonianLoss(H; h = 1e300)(dec32, z32, out32)
    @test_throws re_step("1.0e-50", "0.0") ReducedHamiltonianLoss(H; λ = 0, h = 1e-50)(
        dec32, z32, out32)
end

# With λ = 0 the loss does not evaluate the Hamiltonian, which only the gradient term needs.
@testset "ReducedHamiltonianLoss with λ = 0 skips the gradient term" begin
    dec = make_decoder(Float64)
    z = randn(2, 5)
    out = targets(dec, z)
    never(X) = error("the Hamiltonian is evaluated")
    @test ReducedHamiltonianLoss(never; λ = 0)(dec, z, out) ==
          ReducedHamiltonianLoss(H; λ = 0)(dec, z, out)
end

@testset "ReducedHamiltonianLoss in $T" for T in REAL_ELTYPES
    test_gradient(T)
    test_step(T)
    test_self_consistency(T)
    test_terms(T)
    test_value(T)
    test_continuity_at_zero_target(T)
    test_duplicated_batch(T)
    test_gradient_scale(T)
    test_tensor_method(T)
    test_one_training_step(T)
end

@testset "ReducedHamiltonianLoss parameter gradient" begin
    test_parameter_gradient()
end
