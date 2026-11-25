using GeometricMachineLearning
using GeometricMachineLearning: params, BiorthogonalManifold
using Zygote: Zygote
using Test
using LinearAlgebra: norm, dot
import Random

include("../helpers/eltypes.jl")
include("../helpers/leaves.jl")

Random.seed!(1234)

# |p|²/2 + cos(q₁) + q₂²/2 on ℝ⁴, non-linear in every coordinate, and its vector field 𝕁∇H.
H(X) = vec(sum(abs2, X[3:4, :]; dims = 1)) ./ 2 .+ cos.(X[1, :]) .+ X[2, :] .^ 2 ./ 2
XH(X) = vcat(X[3:4, :], sin.(X[1:1, :]), -X[2:2, :])

const ARCH = SymplecticAutoencoder(4, 2; n_encoder_blocks = 2, n_decoder_blocks = 2)
# its encoder and decoder share one parameter tree, which the loss reverses for the encoder
const LEFT_INVERSE = LeftInverseSymplecticAutoencoder(
    4, 2; n_blocks = 2, n_layers = 3, n_ambient_layers = 2)
# with biorthogonal lifts, which `make_network` moves to a point drawn by `rand`, so that Φ ≠ Ψ
const BIORTHOGONAL = LeftInverseSymplecticAutoencoder(
    4, 2; n_blocks = 2, n_layers = 3, n_ambient_layers = 2, biorthogonal = true)
const ARCHITECTURES = (ARCH, LEFT_INVERSE, BIORTHOGONAL)
const X_REF = [0.3, -0.2, 0.1, 0.4]

function make_network(arch, ::Type{T}) where {T}
    nn = NeuralNetwork(arch, T)
    foreach_leaf(params(nn)) do p
        p isa BiorthogonalManifold && copyto!(p, rand(BiorthogonalManifold{T}, size(p)...))
    end
    nn
end

function label(arch)
    arch === BIORTHOGONAL ? "biorthogonal LeftInverseSymplecticAutoencoder" :
    string(nameof(typeof(arch)))
end

# The references take every derivative by Zygote, one state at a time: the encoder's Jacobian times
# the vector field, and the gradient of H∘Ψ^dec at the encoded state.
function pushed_forward(enc, x, v)
    reduce(hcat, [Zygote.jacobian(enc, x[:, i])[1] * v[:, i] for i in axes(x, 2)])
end
function reduced_field(dec, ξ)
    g = reduce(hcat,
        [Zygote.gradient(ζ -> H(reshape(dec(ζ), 4, 1))[1], ξ[:, i])[1] for i in axes(ξ, 2)])
    vcat(g[2:2, :], -g[1:1, :])
end

# The three terms of the loss before their weights and scales: the relative reconstruction error,
# the norm of the energy error against the reference state, and the norm of the vector-field error
# with the norm of each of the two fields.
function terms(nn, x, v; H = H)
    enc, dec = encoder(nn), decoder(nn)
    T = eltype(x)
    ξ = enc(x)
    d(y) = dec(enc(y))
    x_ref = reshape(T.(X_REF), 4, 1)
    R = norm(d(x) - x) / norm(x)
    E = norm((H(d(x)) .- H(d(x_ref))) .- (H(x) .- H(x_ref)))
    reduced, pushed = reduced_field(dec, ξ), pushed_forward(enc, x, v)
    R, E, norm(reduced - pushed), norm(reduced) + norm(pushed)
end

# The central differences with the default step cbrt(eps(T)) have a truncation and a round-off
# error of order eps(T)^(2/3) each, relative to the size of the fields.
fd_rtol(::Type{T}) where {T} = 20 * cbrt(eps(T))^2

# The loss is the relative reconstruction error, plus λ times the root-mean-square error of the
# latent vector field over the batch divided by vector_field_scale, plus the root-mean-square
# energy error against the reference state divided by energy_scale, for each combination of the
# two flags. The central differences add at most λ * fd_rtol * (‖reduced‖ + ‖pushed‖) / (√B * s).
function test_value(arch, ::Type{T}) where {T}
    nn = make_network(arch, T)
    B = 30
    x = randn(T, 4, B)
    v = XH(x)
    R, E, V, size_V = terms(nn, x, v)
    λ, s, s_E = 0.7, 2.5, 1.5
    for (reconstruction, energy) in Iterators.product((false, true), (false, true))
        loss = LatentVectorFieldLoss(arch, H; λ = λ, vector_field_scale = s,
            reconstruction = reconstruction, energy = energy, reference_state = X_REF,
            energy_scale = s_E)
        l = @inferred loss(nn.model, params(nn), x, v)
        reference = reconstruction * R + energy * E / (sqrt(T(B)) * T(s_E)) +
                    T(λ) * V / (sqrt(T(B)) * T(s))
        @test l isa T
        @test l ≈ reference atol = T(λ) * fd_rtol(T) * size_V / (sqrt(T(B)) * T(s)) +
                                   8 * eps(T) * reference
    end
end

# A given step reaches both central differences: their error is of order h², so halving a large
# step divides the error of the vector-field term by four. The default step is cbrt(eps(T)).
function test_step(arch, ::Type{T}) where {T}
    nn = make_network(arch, T)
    B = 30
    x = randn(T, 4, B)
    v = XH(x)
    _, _, V, _ = terms(nn, x, v)
    l(h) = LatentVectorFieldLoss(arch, H; h = h, reconstruction = false)(nn, x, v)
    err(h) = abs(l(h) - V / sqrt(T(B)))
    @test err(0.2) / err(0.1) ≈ 4 rtol = 0.1
    @test LatentVectorFieldLoss(arch, H; reconstruction = false)(nn, x, v) ≈ l(cbrt(eps(T))) rtol = 8 *
                                                                                                    eps(T)
end

# The vector-field term is continuous in the vector field, also where it vanishes: for X_H = σ·1,
# the triangle inequality bounds the change from σ = 0 by λ/(√B s) times the norm of the pushed-
# forward field, σ‖∇Ψ^enc 1‖ plus the round-off of the central difference, eps(T)‖ξ‖/h.
function test_continuity_at_zero_target(arch, ::Type{T}) where {T}
    nn = make_network(arch, T)
    B = 10
    x = randn(T, 4, B)
    λ, s = 0.7, 2.5
    loss = LatentVectorFieldLoss(arch, H; λ = λ, vector_field_scale = s)
    l(σ) = loss(nn, x, fill(T(σ), 4, B))
    l0 = l(0)
    @test l0 isa T
    @test isfinite(l0) && l0 > 0
    J1 = norm(pushed_forward(encoder(nn), x, ones(T, 4, B)))
    roundoff = 10 * eps(T) * norm(encoder(nn)(x)) / cbrt(eps(T))
    for σ in (1e-6, 1e-12)
        @test abs(l(σ) - l0) <=
              T(λ) / (sqrt(T(B)) * T(s)) * (T(σ) * J1 + roundoff) + 8 * eps(T) * l0
    end
end

# The value does not depend on the size of the batch: the batch [x x] has the loss of x.
function test_duplicated_batch(arch, ::Type{T}) where {T}
    nn = make_network(arch, T)
    x = randn(T, 4, 30)
    v = XH(x)
    loss = LatentVectorFieldLoss(arch, H; λ = 0.7, vector_field_scale = 2.5, energy = true,
        reference_state = X_REF, energy_scale = 1.5)
    @test loss(nn, hcat(x, x), hcat(v, v)) ≈ loss(nn, x, v) rtol = 8 * eps(T)
end

# The loss does not depend on the additive constant of H. The constant c enters the round-off of
# the central differences of H∘Ψ^dec, eps(T)(|H| + c)/h per entry, and that of the energy
# differences, eps(T)(|H| + c) per entry.
function test_additive_constant(arch, ::Type{T}) where {T}
    nn = make_network(arch, T)
    B = 30
    x = randn(T, 4, B)
    v = XH(x)
    λ, s, s_E = 0.7, 2.5, 1.5
    loss(H) = LatentVectorFieldLoss(arch, H; λ = λ, vector_field_scale = s, energy = true,
        reference_state = X_REF, energy_scale = s_E)
    l = loss(H)(nn, x, v)
    H_max = maximum(abs, H(x))
    for c in (-7.0, 100.0)
        e = 10 * eps(T) * (H_max + T(abs(c)))
        @test loss(X -> H(X) .+ T(c))(nn, x, v) ≈ l atol = T(λ) / T(s) * 4 * e /
                                                           cbrt(eps(T)) +
                                                           4 * e / T(s_E)
    end
end

# λ, vector_field_scale and energy_scale weigh their own term and nothing else: the loss is
# l0 + (λ / s) (lV - l0) + (lE - l0) / s_E for the loss l0 with neither term, lV with the
# vector-field term at λ = s = 1 and lE with the energy term at s_E = 1.
function test_scales(arch, ::Type{T}) where {T}
    nn = make_network(arch, T)
    x = randn(T, 4, 30)
    v = XH(x)
    loss(λ, s, energy, s_E) = LatentVectorFieldLoss(arch, H; λ = λ, vector_field_scale = s,
        energy = energy, reference_state = X_REF, energy_scale = s_E)(nn, x, v)
    l0 = loss(0, 1, false, 1)
    lV = loss(1, 1, false, 1)
    lE = loss(0, 1, true, 1)
    @test lV - l0 > 10 * fd_rtol(T)
    @test lE - l0 > 10 * fd_rtol(T)
    for (λ, s, s_E) in Iterators.product((0.7, 3.0), (0.4, 2.5), (0.5, 2.0))
        @test loss(λ, s, true, s_E) ≈ l0 + T(λ) / T(s) * (lV - l0) + (lE - l0) / T(s_E) rtol = 8 *
                                                                                               eps(T)
    end
end

# The 3-tensors of a `DataLoader` give the loss of the matrices.
function test_tensor_method(arch, ::Type{T}) where {T}
    nn = make_network(arch, T)
    x = randn(T, 4, 30)
    v = XH(x)
    loss = LatentVectorFieldLoss(arch, H; λ = 0.7, energy = true, reference_state = X_REF)
    @test loss(nn.model, params(nn), reshape(x, 4, 30, 1), reshape(v, 4, 30, 1)) ≈
          loss(nn.model, params(nn), x, v) rtol = 8 * eps(T)
end

# One `train!` step on a `DataLoader` of the states and the vector field gives a finite loss of the
# network's element type and moves every parameter.
function test_one_training_step(arch, ::Type{T}) where {T}
    nn = make_network(arch, T)
    x = randn(T, 4, 40)
    dl = DataLoader(x, XH(x); suppress_info = true)
    before = deepcopy(nn.params)
    o = TrainingOptimizer(nn.params; algorithm = Adam())
    loss = LatentVectorFieldLoss(arch, H; energy = true, reference_state = X_REF)
    history = train!(nn, o, dl, Batch(40), 1; loss = loss, show_progress = false)
    @test eltype(history) === T
    @test all(isfinite, history)
    moved = Bool[]
    foreach_leaf((p, q) -> push!(moved, p != q), nn.params, before)
    @test all(moved)
end

# The Zygote gradient with respect to the parameters agrees with a central difference of the loss
# in a random direction of the array leaves. A biorthogonal lift is a function of its stored matrix
# [Φ Ψ], whose dense cotangent the gradient holds, so its storage is a leaf too.
function test_parameter_gradient(arch)
    nn = make_network(arch, Float64)
    x = randn(4, 30)
    v = XH(x)
    loss = LatentVectorFieldLoss(arch, H; λ = 0.7, energy = true, reference_state = X_REF)
    ps = params(nn)
    f(p) = loss(nn.model, p, x, v)
    dp = Zygote.gradient(f, ps)[1]
    leaves = Tuple{Array{Float64}, Any, Array{Float64}}[]
    foreach_leaf(ps, dp) do p, g
        p isa BiorthogonalManifold && (p = parent(p))
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
    # error from the round-off of the loss's own central differences, of order eps / h with
    # h = cbrt(eps), divided by ε; the loss is of order one
    @test ad ≈ fd rtol=10*ε^2 atol=10*eps()/(cbrt(eps())*ε)
end

@testset "LatentVectorFieldLoss rejects invalid arguments, networks and outputs" begin
    prefix = "^ArgumentError: LatentVectorFieldLoss "
    re(s) = Regex(prefix * s * "\$")
    re_λ(s) = re("requires a finite weight λ >= 0, got λ = $s\\.")
    @test_throws re_λ("-1") LatentVectorFieldLoss(ARCH, H; λ = -1)
    @test_throws re_λ("NaN") LatentVectorFieldLoss(ARCH, H; λ = NaN)
    @test_throws re_λ("Inf") LatentVectorFieldLoss(ARCH, H; λ = Inf)
    re_h(s) = re("requires a finite step h > 0, got h = $s\\.")
    @test_throws re_h("0") LatentVectorFieldLoss(ARCH, H; h = 0)
    @test_throws re_h("-0.0001") LatentVectorFieldLoss(ARCH, H; h = -1e-4)
    @test_throws re_h("NaN") LatentVectorFieldLoss(ARCH, H; h = NaN)
    @test_throws re_h("Inf") LatentVectorFieldLoss(ARCH, H; h = Inf)
    re_s(s) = re("requires a finite vector_field_scale > 0, got vector_field_scale = $s\\.")
    @test_throws re_s("0") LatentVectorFieldLoss(ARCH, H; vector_field_scale = 0)
    @test_throws re_s("-1.0") LatentVectorFieldLoss(ARCH, H; vector_field_scale = -1.0)
    @test_throws re_s("NaN") LatentVectorFieldLoss(ARCH, H; vector_field_scale = NaN)
    @test_throws re_s("Inf") LatentVectorFieldLoss(ARCH, H; vector_field_scale = Inf)
    re_e(s) = re("requires a finite energy_scale > 0, got energy_scale = $s\\.")
    @test_throws re_e("0") LatentVectorFieldLoss(ARCH, H; energy_scale = 0)
    @test_throws re_e("-1.0") LatentVectorFieldLoss(ARCH, H; energy_scale = -1.0)
    @test_throws re_e("NaN") LatentVectorFieldLoss(ARCH, H; energy_scale = NaN)
    @test_throws re_e("Inf") LatentVectorFieldLoss(ARCH, H; energy_scale = Inf)
    @test_throws re("with energy = true requires a reference_state, the full state x_ref at which the energies are compared\\.") LatentVectorFieldLoss(
        ARCH, H; energy = true)
    re_ref(n) = re("requires a reference_state of length 4, the full dimension of the architecture, got length $n\\.")
    @test_throws re_ref(2) LatentVectorFieldLoss(ARCH, H; energy = true, reference_state = [
        1.0, 2.0])
    @test_throws re_ref(6) LatentVectorFieldLoss(ARCH, H; reference_state = zeros(6))
    @test_throws re("requires a reference_state with finite entries\\.") LatentVectorFieldLoss(
        ARCH, H; energy = true, reference_state = [0.0, NaN, 0.0, 0.0])
    @test LatentVectorFieldLoss(ARCH, H; λ = 0).λ == 0
    @test LatentVectorFieldLoss(ARCH, H; vector_field_scale = 2).vector_field_scale === 2.0
    @test LatentVectorFieldLoss(ARCH, H; energy_scale = 2).energy_scale === 2.0
    @test LatentVectorFieldLoss(ARCH, H).h === nothing
    @test LatentVectorFieldLoss(ARCH, H; energy = true, reference_state = [1, 2, 3, 4]).reference_state ==
          [1.0, 2.0, 3.0, 4.0]
    # the guards cannot be bypassed by a positional constructor
    @test_throws r"^MethodError: no method matching \(?(GeometricMachineLearning\.)?LatentVectorFieldLoss\{3\}\)?\(" LatentVectorFieldLoss{3}(
        H, -1.0, 0.0, 0.0, 0.0, true, false, 5, 2, nothing)

    # the architecture: a symplectic autoencoder whose encoder and decoder the loss can split
    re_arch(name) = re("takes a SymplecticAutoencoder, a PSDArch or a LeftInverseSymplecticAutoencoder, got a $name\\.")
    symmetric = SymmetricSymplecticAutoencoder(2, 1, 2)
    @test_throws re_arch("SymmetricSymplecticAutoencoder") LatentVectorFieldLoss(symmetric, H)
    # the architecture is checked before the reference state, which is measured against it
    @test_throws re_arch("SymmetricSymplecticAutoencoder") LatentVectorFieldLoss(
        symmetric, H; reference_state = zeros(symmetric.full_dim + 2))
    @test_throws r"^MethodError: no method matching (GeometricMachineLearning\.)?LatentVectorFieldLoss\(::(GeometricMachineLearning\.)?StandardAutoencoder" LatentVectorFieldLoss(
        StandardAutoencoder(4, 2), H)

    # the network and the output
    nn = make_network(ARCH, Float64)
    x = randn(4, 5)
    loss = LatentVectorFieldLoss(ARCH, H)
    re_out(s) = re("requires an output of the size of the input, the vector field X_H\\(x\\) at each state x, got an output of size $s for an input of size \\(4, 5\\)\\.")
    @test_throws re_out("\\(4, 1\\)") loss(nn, x, XH(x)[:, 1:1])
    @test_throws re_out("\\(1, 5\\)") loss(nn, x, XH(x)[1:1, :])
    @test_throws re_out("\\(6, 5\\)") loss(nn, x, vcat(XH(x), zeros(2, 5)))
    other = SymplecticAutoencoder(
        4, 2; n_encoder_layers = 2, n_encoder_blocks = 2, n_decoder_blocks = 2)
    function re_layers(n, m)
        re("requires a network of $n layers, those of its architecture, got $m layers; " *
           "build the loss from the architecture of the network\\.")
    end
    @test_throws re_layers(5, 7) LatentVectorFieldLoss(other, H)(nn, x, XH(x))
    # a left-inverse network has the 6 + 6 layers of its encoder and its decoder, and 5 + 5 with
    # n_layers = 2; the network of ARCH has 7
    tied = make_network(LEFT_INVERSE, Float64)
    @test_throws re_layers(12, 7) LatentVectorFieldLoss(LEFT_INVERSE, H)(nn, x, XH(x))
    @test_throws re_layers(7, 12) loss(tied, x, XH(x))
    shorter = LeftInverseSymplecticAutoencoder(
        4, 2; n_blocks = 2, n_layers = 2, n_ambient_layers = 2)
    @test_throws re_layers(10, 12) LatentVectorFieldLoss(shorter, H)(tied, x, XH(x))
    re_latent(n) = re("requires an encoder with 2 output rows, the latent dimension of its architecture, got $n rows; build the loss from the architecture of the network\\.")
    @test_throws re_latent(4) LatentVectorFieldLoss(PSDArch(6, 2), H)(
        NeuralNetwork(PSDArch(6, 4)), randn(6, 5), randn(6, 5))
    # LeftInverseSymplecticAutoencoder(6, 2) and (6, 4) have the same number of layers
    @test_throws re_latent(4) LatentVectorFieldLoss(LeftInverseSymplecticAutoencoder(6, 2), H)(
        NeuralNetwork(LeftInverseSymplecticAutoencoder(6, 4)), randn(6, 5), randn(6, 5))
end

# With λ = 0 and no energy term the loss does not evaluate the Hamiltonian, which only those two
# terms need.
@testset "LatentVectorFieldLoss with λ = 0 skips the vector-field term" begin
    nn = make_network(ARCH, Float64)
    x = randn(4, 5)
    never(X) = error("the Hamiltonian is evaluated")
    @test LatentVectorFieldLoss(ARCH, never; λ = 0)(nn, x, XH(x)) ==
          LatentVectorFieldLoss(ARCH, H; λ = 0)(nn, x, XH(x))
end

@testset "LatentVectorFieldLoss on a $(label(arch)) in $T" for arch in ARCHITECTURES,
    T in REAL_ELTYPES

    test_value(arch, T)
    test_step(arch, T)
    test_continuity_at_zero_target(arch, T)
    test_duplicated_batch(arch, T)
    test_additive_constant(arch, T)
    test_scales(arch, T)
    test_tensor_method(arch, T)
    test_one_training_step(arch, T)
end

@testset "LatentVectorFieldLoss parameter gradient on a $(label(arch))" for arch in ARCHITECTURES
    test_parameter_gradient(arch)
end
