# `VolumePreservingAttention` at every sequence length: the forward pass against an independent host
# computation of the same transform, and the gradient against the gradient on `Array`. The layer's
# own `Array` result is not the reference, because the device and the host run the same kernel.
#
# The host reference builds `C` with plain products and solves `(I + C) \ (I - C)` per slice; the
# tolerance of the forward pass is the bound of the Cayley kernel's value row, `10 n eps(T) (1 +
# ‖Cₖ‖₂²)`, times the size of `Z`, which is what the product that follows the transform scales it by.

using GeometricMachineLearning
using GeometricMachineLearning: tensor_mat_skew_sym_assign, _mul
using FiniteDifferences: central_fdm, j′vp
using GPUArraysCore: allowscalar
using JLArrays: JLArray, JLBackend
import KernelAbstractions
using LinearAlgebra: I, dot, opnorm
using Zygote: gradient
using Test
import Random

Random.seed!(1234)
allowscalar(false)

ondevice(x) = KernelAbstractions.get_backend(x) isa JLBackend

function cayley_reference(C::AbstractMatrix{T}) where {T}
    Id = Matrix{T}(I, size(C))
    (Id + C) \ (Id - C)
end

# The `:arbitrary` path's `C`, slice by slice: the lower triangle of the scalar products of the
# columns of `Z` weighted by `A`, divided by `√M`, and made skew by subtracting its own transpose —
# the same triangle, the same divisor and the same subtraction the layer makes.
function arbitrary_C(x::AbstractArray{T, 3}, A::AbstractMatrix) where {T}
    M, seq = size(x, 1), size(x, 2)
    C = zeros(T, seq, seq, size(x, 3))
    for k in axes(x, 3)
        X = Matrix(@view x[:, :, k])
        L = zeros(T, seq, seq)
        for i in 1:seq, j in 1:(i - 1)

            L[i, j] = dot(@view(X[:, i]), A * @view(X[:, j]))
        end
        C[:, :, k] = (L - L') / T(√M)
    end
    C
end

function skew_sym_C(x::AbstractArray{T, 3}, A) where {T}
    C = zeros(T, size(x, 2), size(x, 2), size(x, 3))
    D = Matrix(A)
    for k in axes(x, 3)
        X = Matrix(@view x[:, :, k])
        C[:, :, k] = X' * (D * X)
    end
    C
end

function C_tensor(d, x, ps)
    d isa VolumePreservingAttention{<:Any, <:Any, :skew_sym} ?
    skew_sym_C(x, ps.A) : arbitrary_C(x, ps.A)
end

# The layer's forward pass, slice by slice on the host: the same `C` the layer forms, a dense solve
# of the Cayley transform, and the product with the data.
function reference_forward(d::VolumePreservingAttention, x, ps)
    C = C_tensor(d, x, ps)
    out = similar(x)
    for k in axes(x, 3)
        out[:, :, k] = Matrix(@view x[:, :, k]) * cayley_reference(@view C[:, :, k])
    end
    out
end

# The bound of the Cayley kernel's value row, scaled by the largest entry of the data: the product
# `Z Λ` multiplies the transform's error by `Z`.
function tolerance(d, x, ps)
    T = eltype(x)
    C = C_tensor(d, x, ps)
    worst = maximum(Float64(opnorm(@view(C[:, :, k]), 2))^2 for k in axes(x, 3))
    10 * size(x, 2) * eps(T) * (1 + worst) * Float64(maximum(abs, Array(x)))
end

# A layer of dimension 4 and sequence length `SL` (`0` for the default, which takes data of any
# length), its parameters and its data, all on the host; a device case converts them, as
# `test/devices/jlarray.jl` does, rather than initialising them on the device.
function case(SL, seq_length, T; skew_sym)
    d = VolumePreservingAttention(4, SL; skew_sym)
    model = Chain(d)
    ps = NeuralNetwork(model, CPU(), T).params.L1
    x = randn(T, 4, seq_length, 3)
    d, ps, x
end

@testset "VolumePreservingAttention runs at every sequence length, T = $T" for T in (Float32, Float64)
    for N in 2:8, skew_sym in (false, true)

        d, ps, x = case(N, N, T; skew_sym)
        device = VolumePreservingAttention(4, N; skew_sym)
        ps_device = (A = skew_sym ? SkewSymMatrix(JLArray(ps.A.S), 4) : JLArray(ps.A),)

        y_device = device(JLArray(x), ps_device)
        @test ondevice(y_device)

        # The forward pass against the independent host computation.
        reference = reference_forward(d, x, ps)
        @test maximum(abs, Array(y_device) - reference) <= tolerance(d, x, ps)

        # The gradient with respect to the data, on the device against the host.
        gx_device = gradient(z -> sum(device(z, ps_device)), JLArray(x))[1]
        gx_host = gradient(z -> sum(d(z, ps)), x)[1]
        @test ondevice(gx_device)
        @test maximum(abs, Array(gx_device) - gx_host) <= tolerance(d, x, ps)

        # The gradient with respect to the weight, on the device against the host.
        gA_device = gradient(A -> sum(device(JLArray(x), (A = A,))), ps_device.A)[1]
        gA_host = gradient(A -> sum(d(x, (A = A,))), ps.A)[1]
        if skew_sym
            @test ondevice(gA_device.S)
            @test maximum(abs, Array(gA_device.S) - gA_host.S) <= tolerance(d, x, ps)
        else
            @test ondevice(gA_device)
            @test maximum(abs, Array(gA_device) - gA_host) <= tolerance(d, x, ps)
        end
    end
end

@testset "VolumePreservingAttention with SL = 0 runs every sequence length, T = $T" for T in (Float32,
    Float64)
    for seq_length in (3, 7), skew_sym in (false, true)

        d, ps, x = case(0, seq_length, T; skew_sym)

        y = d(x, ps)
        @test size(y) == size(x)
        @test maximum(abs, y - reference_forward(d, x, ps)) <= tolerance(d, x, ps)

        ps_device = (A = skew_sym ? SkewSymMatrix(JLArray(ps.A.S), 4) : JLArray(ps.A),)
        y_device = d(JLArray(x), ps_device)
        @test ondevice(y_device)
        @test maximum(abs, Array(y_device) - reference_forward(d, x, ps)) <=
              tolerance(d, x, ps)

        # The default layer has no symbolic inverse, so its pullback runs the kernel's; both
        # gradients are compared with the host's, as at a fixed sequence length.
        gx_device = gradient(z -> sum(d(z, ps_device)), JLArray(x))[1]
        gx_host = gradient(z -> sum(d(z, ps)), x)[1]
        @test ondevice(gx_device)
        @test maximum(abs, Array(gx_device) - gx_host) <= tolerance(d, x, ps)

        gA_device = gradient(A -> sum(d(JLArray(x), (A = A,))), ps_device.A)[1]
        gA_host = gradient(A -> sum(d(x, (A = A,))), ps.A)[1]
        if skew_sym
            @test ondevice(gA_device.S)
            @test maximum(abs, Array(gA_device.S) - gA_host.S) <= tolerance(d, x, ps)
        else
            @test ondevice(gA_device)
            @test maximum(abs, Array(gA_device) - gA_host) <= tolerance(d, x, ps)
        end
    end
end

@testset "the gradient of VolumePreservingAttention at N = 6 against finite differences" begin
    T = Float64
    d, ps, x = case(6, 6, T; skew_sym = false)
    Λ̄ = randn(T, size(x))
    fd = j′vp(central_fdm(5, 1), z -> d(z, ps), Λ̄, x)[1]
    analytic = gradient(z -> sum(Λ̄ .* d(z, ps)), x)[1]
    # `1e-8` in `Float64`, as `test/integration/product_gradients.jl` measures the products.
    @test isapprox(analytic, fd; rtol = 1e-8)
end
