using GeometricMachineLearning
using KernelAbstractions: KernelAbstractions, CPU
using JET
using Test

include("../helpers/eltypes.jl")

const GML = GeometricMachineLearning
const TARGET = (GML,)
const rrule = GML.ChainRulesCore.rrule

# The entry points are the functions that launch a kernel of `src/` on KernelAbstractions' `CPU()`
# backend, the pullbacks among them. A launcher line analyses one entry point at the argument types
# of one element type. Where a test outside `test/quality/` calls the entry point directly, the
# element types are those of the direct calls; an element type that reaches the entry point only
# through another function gets no line. Where no test
# calls the entry point directly, the element types are those at which the tests reach it through
# its callers. Another container type at the same element type gets no line.
#
# JET drops the reports of a kernel body when it analyses the launcher, so each `@kernel` method
# also gets a line per element type of its launchers, which analyses the generated
# `cpu_<kernel>` function directly, at the `CompilerMetadata` context that the launcher builds.
# This uses internals of KernelAbstractions (`launch_config`, `mkcontext`, `blocks`, `Kernel.f`).
# These lines filter with `JET.AnyFrameModule(GML)`, so that a dispatch in a function that the
# kernel body calls counts.
# One kind of dispatch stays unseen, `KNOWN_ISSUES.md` K3: a type-unstable argument of a kernel
# launch, because its dispatch sits in KernelAbstractions' frames. JET never reports a dynamic
# dispatch in a kernel's own frame, and a barrier on a value inside a kernel body is reported.
function kernel_body_reports(kernel, ndrange, args...)
    k = kernel(CPU())
    nd, _, iterspace, dynamic = KernelAbstractions.launch_config(k, ndrange, nothing)
    ctx = KernelAbstractions.mkcontext(
        k, first(KernelAbstractions.blocks(iterspace)), nd, iterspace, dynamic)
    JET.get_reports(JET.report_opt(k.f, (typeof(ctx), map(typeof, args)...);
        target_modules = (JET.AnyFrameModule(GML),)))
end

const A3{T} = Array{T, 3}

@testset "JET" begin
    if isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE
        # src/kernels/tensor_tensor_mul.jl and the two transposed products
        for T in (Float16, Float32, Float64)
            @test isempty(JET.get_reports(JET.report_opt(GML.tensor_tensor_mul!,
                (A3{T}, A3{T}, A3{T}); target_modules = TARGET)))
            @test isempty(kernel_body_reports(GML.tensor_tensor_mul_kernel!, (2, 3, 2),
                zeros(T, 2, 3, 2), zeros(T, 2, 4, 2), zeros(T, 4, 3, 2)))
            @test isempty(JET.get_reports(JET.report_opt(GML.tensor_transpose_tensor_mul!,
                (A3{T}, A3{T}, A3{T}); target_modules = TARGET)))
            @test isempty(kernel_body_reports(GML.tensor_transpose_tensor_mul_kernel!,
                (2, 3, 2), zeros(T, 2, 3, 2), zeros(T, 4, 2, 2), zeros(T, 4, 3, 2)))
        end
        for T in REAL_ELTYPES
            @test isempty(JET.get_reports(JET.report_opt(GML.tensor_tensor_transpose_mul!,
                (A3{T}, A3{T}, A3{T}); target_modules = TARGET)))
            @test isempty(kernel_body_reports(GML.tensor_tensor_transpose_mul_kernel!,
                (2, 3, 2), zeros(T, 2, 3, 2), zeros(T, 2, 4, 2), zeros(T, 3, 4, 2)))
        end

        # src/kernels/cayley.jl: `tensor_cayley`, its kernel and its pullback
        for T in (Float16, Float32, Float64)
            @test isempty(JET.get_reports(JET.report_opt(GML.tensor_cayley, (A3{T},);
                target_modules = TARGET)))
            @test isempty(kernel_body_reports(GML.tensor_cayley_kernel!, 2,
                zeros(T, 3, 3, 2), zeros(T, 3, 3, 2), zeros(T, 3, 3, 2)))
            @test isempty(JET.get_reports(JET.report_opt(
                rrule(GML.tensor_cayley, zeros(T, 3, 3, 2))[2], (A3{T},);
                target_modules = TARGET)))
        end

        # src/kernels/kernel_ad_routines/: the pullback that launches kernels, at the `Float64`
        # types of `test_rrule` in test/kernels/kernel_pullbacks.jl
        Z, M = zeros(3, 4, 2), zeros(3, 3)
        @test isempty(JET.get_reports(JET.report_opt(
            rrule(GML.tensor_mat_skew_sym_assign, Z, M)[2], (A3{Float64},);
            target_modules = TARGET)))
        @test isempty(kernel_body_reports(GML.dz_kernel!, size(Z), zero(Z), Z, M,
            zeros(4, 4, 2), 3, 4))
        @test isempty(kernel_body_reports(GML.da_kernel!, (3, 3, 2), zeros(3, 3, 2), Z, M,
            zeros(4, 4, 2), 3, 4))

        # src/kernels/inverses/tensor_mat_skew_sym_assign.jl
        for T in (Float16, Float32, Float64)
            @test isempty(JET.get_reports(JET.report_opt(GML.tensor_mat_skew_sym_assign!,
                (A3{T}, A3{T}, Matrix{T}); target_modules = TARGET)))
            @test isempty(kernel_body_reports(GML.tensor_mat_skew_sym_assign_kernel!,
                (4, 4, 2), zeros(T, 4, 4, 2), zeros(T, 3, 4, 2), zeros(T, 3, 3)))
        end
    else
        @test_skip "JET does not work on Julia $(VERSION)"  # aviatesk/JET.jl#681
    end
end
