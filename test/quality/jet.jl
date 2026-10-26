using GeometricMachineLearning
using KernelAbstractions: KernelAbstractions, CPU
using JET
using Test

include("../helpers/eltypes.jl")

const GML = GeometricMachineLearning
const TARGET = (GML,)
const rrule = GML.ChainRulesCore.rrule

# The entry points are the functions that launch a kernel of `src/` on KernelAbstractions' `CPU()`
# backend, the pullbacks among them, and the `Batch` functor that
# `test/data_loader/batch_index_set.jl` asserts with `@allocated`. A launcher line analyses one
# entry point at the argument types of one element type. Where a test outside `test/quality/`
# calls the entry point directly, the element types are those of the direct calls; an element
# type that reaches the entry point only through another function gets no line. Where no test
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

        # src/data_loader/batch.jl
        indices = ones(Int, 2, 4)
        TimeSeriesQP = DataLoader{
            Float64, @NamedTuple{q::A3{Float64}, p::A3{Float64}}, Nothing, :TimeSeries, Nothing}
        Indices = Vector{Tuple{Int, Int}}
        # test/data_loader/draw_batch_for_tensor_test.jl
        @test isempty(JET.get_reports(JET.report_opt(
            GML.convert_input_and_batch_indices_to_array,
            (TimeSeriesQP, Batch{:Transformer}, Indices); target_modules = TARGET)))
        # the one gather kernel of a `(q, p)` input: `gather` launches it once per tensor
        @test isempty(kernel_body_reports(GML.gather_kernel!,
            (2, 3, 4), zeros(2, 3, 4), zeros(2, 8, 4), indices, 3))
        # test/integration/docstrings/data_loader.jl
        @test isempty(JET.get_reports(JET.report_opt(
            GML.convert_input_and_batch_indices_to_array,
            (DataLoader{Float64, A3{Float64}, Nothing, :RegularData, Nothing},
                Batch{:FeedForward},
                Indices); target_modules = TARGET)))
        for (T, BatchType) in ((Float32, Batch{:Transformer}), (
            Float64, Batch{:FeedForward}))
            # the optimizer functor in test/data_loader/training_history_eltype.jl
            @test isempty(JET.get_reports(JET.report_opt(
                GML.convert_input_and_batch_indices_to_array,
                (DataLoader{T, A3{T}, Nothing, :TimeSeries, Nothing},
                    Batch{:FeedForward},
                    Indices);
                target_modules = TARGET)))
            # input and output, of kind `:TimeSeries`: no loader of the suite has this type.
            # `DataLoader(input, output)` builds `:RegularData`, and each `:TimeSeries` loader
            # with an output that a test builds by naming the type has an `Array{Int, 3}` output
            @test isempty(JET.get_reports(JET.report_opt(
                GML.convert_input_and_batch_indices_to_array,
                (DataLoader{T, A3{T}, A3{T}, :TimeSeries, Int}, BatchType, Indices);
                target_modules = TARGET)))
            # the gather kernel, once per element type
            @test isempty(kernel_body_reports(GML.gather_kernel!,
                (2, 3, 4), zeros(T, 2, 3, 4), zeros(T, 2, 8, 4), indices, 3))
        end
        # the `Batch` functor under `@allocated` in test/data_loader/batch_index_set.jl, and
        # the other element types of its direct calls: `Int` in
        # test/integration/docstrings/data_loader.jl, `Float64` in
        # test/data_loader/draw_batch_for_tensor_test.jl
        @test isempty(JET.get_reports(JET.report_opt(Batch(8),
            (DataLoader{Float32, A3{Float32}, Nothing, :TimeSeries, Nothing},);
            target_modules = TARGET)))
        @test isempty(JET.get_reports(JET.report_opt(Batch(3),
            (DataLoader{Int, A3{Int}, Nothing, :TimeSeries, Nothing},);
            target_modules = TARGET)))
        @test isempty(JET.get_reports(JET.report_opt(Batch(10, 16), (TimeSeriesQP,);
            target_modules = TARGET)))
        # `Batch(batch_size, dl)` reads the loader's own two lengths, and
        # test/data_loader/batch.jl calls it at the element types below
        @test isempty(JET.get_reports(JET.report_opt(Batch(2),
            (DataLoader{Float32, A3{Float32}, A3{Int}, :RegularData, Int},);
            target_modules = TARGET)))
        @test isempty(JET.get_reports(JET.report_opt(Batch(2),
            (DataLoader{Float32, A3{Float32}, A3{Float32}, :TimeSeries, Int},);
            target_modules = TARGET)))

        # src/data_loader/tensor_assign.jl: `assign_output_estimate` in
        # test/data_loader/accuracy.jl, in both precisions
        for T in REAL_ELTYPES
            @test isempty(JET.get_reports(JET.report_opt(GML.assign_output_estimate,
                (A3{T}, Int); target_modules = TARGET)))
        end
    else
        @test_skip "JET does not work on Julia $(VERSION)"  # aviatesk/JET.jl#681
    end
end
