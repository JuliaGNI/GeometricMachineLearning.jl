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
using Metal
using Test
import Random

# The bound `10 n eps(T) (1 + ‖Cₖ‖₂²)` of the kernel's orthogonality and value rows, the dense
# reference `cayley_reference`, the skew batch `skew_batch`, and `arbitrary_C`, the host form of the
# layer's `C`.
include("../helpers/cayley.jl")

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
            C = skew_batch(T, n, 8)
            Λ = tensor_cayley(MtlArray(C))
            @test KernelAbstractions.get_backend(Λ) == backend
            @test maximum(abs, Array(Λ) - cayley_reference(C)) <= cayley_bound(C)

            # The layer's forward pass against the host: the same `C`, a dense solve of the
            # transform, and the product with the data, at the bound scaled by the data.
            d = VolumePreservingAttention(n, n)
            ps = NeuralNetwork(Chain(d), CPU(), T).params.L1
            x = rand(T, n, n, 8)
            y = d(MtlArray(x), (A = MtlArray(ps.A),))
            C_layer = arbitrary_C(x, ps.A)
            expected = stack(k -> x[:, :, k] * cayley_reference(C_layer[:, :, k]), axes(x, 3))
            @test KernelAbstractions.get_backend(y) == backend
            @test maximum(abs, Array(y) - expected) <=
                  cayley_bound(C_layer) * maximum(abs, x)
        end
    end
else
    @test_skip Metal.functional()
end
