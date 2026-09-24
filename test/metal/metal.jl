# What this package does on a real Apple GPU. `runtests.jl` includes this directory on every
# Apple-silicon Mac.
#
# Where `Metal.functional()` is `false` the file skips itself: on a Mac without a usable device, and
# inside a sandbox, where `Metal.device()` can be a null device although the hardware is present.
# `Pkg.test(test_args = ["metal"])` asks for this directory alone, and then a missing device is a
# failure rather than a skip. `.github/workflows/Metal.yml` runs it that way, so that job cannot
# pass without having run these tests.
#
# `Float32` throughout, because Metal has no `Float64`. Scalar indexing is turned off, so a product
# that reads an array one element at a time raises instead of answering slowly.

using GeometricMachineLearning
using GeometricMachineLearning: tensor_mat_mul!
import KernelAbstractions
using Metal
using Test
import Random

if Metal.functional()
    Metal.allowscalar(false)
    Random.seed!(123)
    @info "Metal device" Metal.device()

    const backend = MetalBackend()
    const T = Float32

    @testset "the GPU extension's wrapped-array products answer on Metal" begin
        # A `SubArray` and an `Adjoint` over an `MtlArray` are not strided, so they miss the fast
        # path of `PoissonTensor`'s `*` and reach `ext/GPUArraysCoreExt.jl`. Without it they fell
        # through to the generic multiply and raised "Scalar indexing is disallowed.".
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

    @testset "tensor_mat_mul! on Metal matches the host" begin
        a = rand(T, 5, 12, 100)
        b = rand(T, 12, 10)
        c = KernelAbstractions.zeros(backend, T, 5, 10, 100)
        tensor_mat_mul!(c, MtlArray(a), MtlArray(b))
        @test all(i -> Array(c)[:, :, i] ≈ a[:, :, i] * b, 1:100)
    end
elseif "metal" in ARGS
    @test Metal.functional()
else
    @info "Metal is not functional here; the Metal tests are skipped."
end
