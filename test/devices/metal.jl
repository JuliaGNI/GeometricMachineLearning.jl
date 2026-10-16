# What this package does on a real Apple GPU. `runtests.jl` includes this file in the `metal` group,
# which a run with no test arguments on an Apple-silicon Mac selects, and which
# `Pkg.test(test_args = ["metal"])` asks for anywhere.
#
# Where `Metal.functional()` is true, the tests below run. Where it is false, the file records one
# skip, `@test_skip Metal.functional()`, and runs no other test: on a Mac without a usable device,
# and inside a sandbox, where `Metal.device()` can be a null device although the hardware is present.
# The skip is visible in the Test Summary as `Broken 1`, and does not fail the run. The guarantee that
# the tests ran is `.github/workflows/Metal.yml`: a step before the tests fails that job where
# `Metal.functional()` is false.
#
# `Float32` throughout, because Metal has no `Float64`. Scalar indexing is turned off, so a product
# that reads an array one element at a time raises instead of answering slowly.

using GeometricMachineLearning
using GeometricMachineLearning: _mul, tensor_tensor_mul, tensor_cayley
import KernelAbstractions
using LinearAlgebra: I, dot, opnorm
using Metal
using Test
import Random

# The layer's forward pass on the host, slice by slice: the same scalar products the kernel computes
# for `C`, a dense solve of the Cayley transform, and the product with the data. It returns the
# slices of `C` as well, because the bound of the kernel's orthogonality rows is written in them.
function host_forward(x::AbstractArray{T, 3}, A::AbstractMatrix) where {T}
    M = size(x, 1)
    out = similar(x)
    C = zeros(T, M, M, size(x, 3))
    for k in axes(x, 3)
        X = Matrix(view(x, :, :, k))
        L = zeros(T, M, M)
        for i in 1:M, j in 1:(i - 1)

            L[i, j] = dot(view(X, :, i), A * view(X, :, j))
        end
        C[:, :, k] = (L - L') / T(√M)
        Id = Matrix{T}(I, M, M)
        out[:, :, k] = X * ((Id + C[:, :, k]) \ (Id - C[:, :, k]))
    end
    out, C
end

if Metal.functional()
    Metal.allowscalar(false)
    Random.seed!(123)
    @info "Metal device" Metal.device()

    const backend = MetalBackend()
    const T = Float32

    @testset "the GPU extension's wrapped-array products answer on Metal" begin
        # A `SubArray` and an `Adjoint` over an `MtlArray` are not strided, so they miss the fast
        # path of `PoissonTensor`'s `*` and reach `ext/GPUArraysCoreExt.jl`. Without it they fall
        # through to the generic multiply and raise "Scalar indexing is disallowed.".
        𝕁 = PoissonTensor(backend, 4, T)
        J = Matrix(PoissonTensor(4, T))
        A = rand(T, 6, 3)
        B = rand(T, 3, 4)

        rows = 𝕁 * view(MtlArray(A), [1, 2, 3, 4], :)
        @test KernelAbstractions.get_backend(rows) == backend
        @test Array(rows) ≈ J * A[1:4, :]

        adjoint_product = 𝕁 * MtlArray(B)'
        @test KernelAbstractions.get_backend(adjoint_product) == backend
        @test Array(adjoint_product) ≈ J * B'
    end

    @testset "_mul and tensor_tensor_mul on Metal match the host" begin
        x = rand(T, 12, 10, 100)
        W = rand(T, 5, 12)
        S = rand(SymmetricMatrix{T}, 12)
        V = rand(T, 10, 4)
        y = rand(T, 10, 7, 100)
        products = ((_mul(MtlArray(W), MtlArray(x)), _mul(W, x)),
            (_mul(SymmetricMatrix(MtlArray(S.S), S.n), MtlArray(x)), _mul(S, x)),
            (_mul(MtlArray(x), MtlArray(V)), _mul(x, V)),
            (tensor_tensor_mul(MtlArray(x), MtlArray(y)), tensor_tensor_mul(x, y)))
        for (device, host) in products
            @test KernelAbstractions.get_backend(device) == backend
            @test Array(device) ≈ host
        end
    end

    # The Cayley kernel and the layer that uses it, at every sequence length: each result stays on
    # the device and matches a host computation of the same transform, at the bound of the kernel's
    # orthogonality rows.
    @testset "tensor_cayley and VolumePreservingAttention on Metal" begin
        for n in 2:8
            R = rand(T, n, n, 8)
            C = R - permutedims(R, (2, 1, 3))
            Λ = tensor_cayley(MtlArray(C))
            @test KernelAbstractions.get_backend(Λ) == backend
            Id = Matrix{T}(I, n, n)
            reference = cat(((Id + C[:, :, k]) \ (Id - C[:, :, k]) for k in axes(C, 3))...;
                dims = 3)
            bound = 10 * n * eps(T) *
                    (1 + maximum(opnorm(C[:, :, k], 2)^2 for k in axes(C, 3)))
            @test maximum(abs, Array(Λ) - reference) <= bound

            d = VolumePreservingAttention(n, n)
            ps = NeuralNetwork(Chain(d), CPU(), T).params.L1
            x = rand(T, n, n, 8)
            y = d(MtlArray(x), (A = MtlArray(ps.A),))
            expected, C_layer = host_forward(x, ps.A)
            layer_bound = 10 * n * eps(T) *
                          (1 + maximum(opnorm(C_layer[:, :, k], 2)^2 for k in axes(x, 3))) *
                          maximum(abs, x)
            @test KernelAbstractions.get_backend(y) == backend
            @test maximum(abs, Array(y) - expected) <= layer_bound
        end
    end
else
    @test_skip Metal.functional()
end
