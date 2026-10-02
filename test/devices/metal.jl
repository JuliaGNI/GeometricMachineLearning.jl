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

    @testset "tensor_mat_mul! on Metal matches the host" begin
        a = rand(T, 5, 12, 100)
        b = rand(T, 12, 10)
        c = KernelAbstractions.zeros(backend, T, 5, 10, 100)
        tensor_mat_mul!(c, MtlArray(a), MtlArray(b))
        @test all(i -> Array(c)[:, :, i] ≈ a[:, :, i] * b, 1:100)
    end
else
    @test_skip Metal.functional()
end
