# The kernels of `src/kernels/` on a device array that has no BLAS and no NNlib extension: a
# `JLArray` with scalar indexing off, so that a kernel that reads one element at a time raises.
# Each result must stay on `JLBackend()` and match the host.

using GeometricMachineLearning
using GeometricMachineLearning: _mul, tensor_tensor_mul, tensor_cayley, assign_q_and_p
using ChainRulesCore: rrule
using GPUArraysCore: allowscalar
using JLArrays: JLArray, JLBackend
using LinearAlgebra: I, opnorm
import KernelAbstractions
using Zygote: gradient
using Test
import Random

Random.seed!(123)
allowscalar(false)

ondevice(x) = KernelAbstractions.get_backend(x) isa JLBackend

@testset "tensor_tensor_mul and its pullback on a JLArray" begin
    for T in (Float32, Float64)
        a, b, c̄ = rand(T, 3, 4, 5), rand(T, 4, 2, 5), rand(T, 3, 2, 5)
        c, pullback = rrule(tensor_tensor_mul, JLArray(a), JLArray(b))
        _, ā, b̄ = pullback(JLArray(c̄))
        _, ā_host, b̄_host = rrule(tensor_tensor_mul, a, b)[2](c̄)
        @test all(ondevice, (c, ā, b̄))
        @test isapprox(Array(c), tensor_tensor_mul(a, b); rtol = sqrt(eps(T)))
        @test isapprox(Array(ā), ā_host; rtol = sqrt(eps(T)))
        @test isapprox(Array(b̄), b̄_host; rtol = sqrt(eps(T)))
    end
end

@testset "_mul with a structured weight on a JLArray: forward and storage gradient" begin
    for T in (Float32, Float64)
        x = rand(T, 4, 6, 3)
        for TT in (SymmetricMatrix, SkewSymMatrix, StrictlyLowerTriangular, StrictlyUpperTriangular)
            W = rand(TT{T}, 4)
            W_device = TT(JLArray(W.S), W.n)
            ps = NetworkParameters((L1 = (W = W,),))
            ps_device = NetworkParameters((L1 = (W = W_device,),))
            y = _mul(W_device, JLArray(x))
            g = gradient(p -> sum(abs2, _mul(p.L1.W, JLArray(x))), ps_device)[1].L1.W
            g_host = gradient(p -> sum(abs2, _mul(p.L1.W, x)), ps)[1].L1.W
            @test ondevice(y)
            @test ondevice(g.S)
            @test isapprox(Array(y), _mul(W, x); rtol = sqrt(eps(T)))
            @test isapprox(Array(g.S), g_host.S; rtol = sqrt(eps(T)))
        end
    end
end

@testset "assign_q_and_p on a JLArray and on a view of one" begin
    for T in (Float32, Float64)
        x = rand(T, 6, 3)
        for v in (JLArray(x), view(JLArray(x), :, 1:2))
            qp = assign_q_and_p(v, 3)
            @test ondevice(qp.q) && ondevice(qp.p)
            @test Array(qp.p) == Array(v)[4:6, :]
        end
    end
end

# The Cayley kernel's own device rows: it launches where scalar indexing is off, its result stays on
# the device, and it matches the per-slice solve on the host at the bound of its orthogonality rows.
@testset "tensor_cayley and its pullback on a JLArray" begin
    for T in (Float32, Float64), n in (2, 3, 8)

        R = rand(T, n, n, 4)
        C = R - permutedims(R, (2, 1, 3))
        Λ, pullback = rrule(tensor_cayley, JLArray(C))
        Λ̄ = rand(T, n, n, 4)
        _, C̄ = pullback(JLArray(Λ̄))
        @test ondevice(Λ) && ondevice(C̄)

        Id = Matrix{T}(I, n, n)
        reference = cat(((Id + C[:, :, k]) \ (Id - C[:, :, k]) for k in axes(C, 3))...; dims = 3)
        bound = 10 * n * eps(T) * (1 + maximum(opnorm(C[:, :, k], 2)^2 for k in axes(C, 3)))
        @test maximum(abs, Array(Λ) - reference) <= bound

        _, C̄_host = rrule(tensor_cayley, C)[2](Λ̄)
        @test maximum(abs, Array(C̄) - C̄_host) <= bound * maximum(abs, C̄_host)
    end
end
