using GeometricMachineLearning
using KernelAbstractions: KernelAbstractions, CPU
using JET
using Test

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
# It filters with `JET.AnyFrameModule(GML)`, so that a dispatch in a function that the kernel
# body calls counts. This uses internals of KernelAbstractions (`launch_config`, `mkcontext`, `blocks`, `Kernel.f`).
# Two kinds of dispatch stay unseen, `KNOWN_ISSUES.md` K3: a type-unstable argument of a kernel
# launch, because its dispatch sits in KernelAbstractions' frames; and, in
# `assign_ones_for_poisson_tensor_kernel!`, an unstable target array `J` or size `n` of its
# splatted store, because JET never reports a dynamic dispatch in the kernel's own frame. A
# barrier on a value inside each kernel body is reported.
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
        # src/kernels/assign_q_and_p.jl: `test_rrule` in test/kernels/kernel_pullbacks.jl
        @test isempty(JET.get_reports(JET.report_opt(GML.assign_q_and_p,
            (Vector{Float64}, Int); target_modules = TARGET)))
        @test isempty(JET.get_reports(JET.report_opt(GML.assign_q_and_p,
            (Matrix{Float64}, Int); target_modules = TARGET)))
        @test isempty(JET.get_reports(JET.report_opt(GML.assign_q_and_p,
            (A3{Float64}, Int); target_modules = TARGET)))
        @test isempty(kernel_body_reports(GML.assign_first_half!, 2, zeros(2), zeros(4)))
        @test isempty(kernel_body_reports(
            GML.assign_second_half!, 2, zeros(2), zeros(4), 2))
        @test isempty(kernel_body_reports(GML.assign_first_half!, (2, 3), zeros(2, 3),
            zeros(4, 3)))
        @test isempty(kernel_body_reports(GML.assign_second_half!, (2, 3), zeros(2, 3),
            zeros(4, 3), 2))
        @test isempty(kernel_body_reports(
            GML.assign_first_half!, (2, 3, 2), zeros(2, 3, 2),
            zeros(4, 3, 2)))
        @test isempty(kernel_body_reports(GML.assign_second_half!, (2, 3, 2),
            zeros(2, 3, 2), zeros(4, 3, 2), 2))

        # src/kernels/tensor_mat_mul.jl: `tensor_mat_mul!` in test/kernels/tensor_mat_mul.jl
        @test isempty(JET.get_reports(JET.report_opt(GML.tensor_mat_mul!,
            (A3{Float32}, A3{Float32}, Matrix{Float32}); target_modules = TARGET)))
        @test isempty(kernel_body_reports(GML.tensor_mat_mul_kernel!, (3, 4, 2),
            zeros(Float32, 3, 4, 2), zeros(Float32, 3, 5, 2), zeros(Float32, 5, 4)))
        # `symmetric_mat_right_mul!` through `tensor_mat_mul!` with a `SymmetricMatrix`:
        # `Float32` in test/architectures/linear_symplectic_transformer.jl, `Float64` in
        # test/kernels/tensor_mat_mul.jl
        for T in (Float32, Float64)
            @test isempty(JET.get_reports(JET.report_opt(GML.symmetric_mat_right_mul!,
                (A3{T}, A3{T}, Vector{T}, Int); target_modules = TARGET)))
            @test isempty(kernel_body_reports(
                GML.symmetric_mat_right_mul_kernel!, (2, 3, 2),
                zeros(T, 2, 3, 2), zeros(T, 2, 3, 2), zeros(T, 6), 3))
        end

        # src/kernels/tensor_tensor_mul.jl and the three transposed products
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
        for T in (Float32, Float64)
            @test isempty(JET.get_reports(JET.report_opt(GML.tensor_tensor_transpose_mul!,
                (A3{T}, A3{T}, A3{T}); target_modules = TARGET)))
            @test isempty(kernel_body_reports(GML.tensor_tensor_transpose_mul_kernel!,
                (2, 3, 2), zeros(T, 2, 3, 2), zeros(T, 2, 4, 2), zeros(T, 3, 4, 2)))
        end
        @test isempty(JET.get_reports(JET.report_opt(GML.tensor_transpose_mat_mul!,
            (A3{Float64}, A3{Float64}, Matrix{Float64}); target_modules = TARGET)))
        @test isempty(kernel_body_reports(GML.tensor_transpose_mat_mul_kernel!, (2, 3, 2),
            zeros(2, 3, 2), zeros(4, 2, 2), zeros(4, 3)))

        # src/kernels/tensor_transpose.jl and src/kernels/exponentials/tensor_exponential.jl
        for T in (Float16, Float32, Float64)
            @test isempty(JET.get_reports(JET.report_opt(GML.tensor_transpose!,
                (A3{T}, A3{T}); target_modules = TARGET)))
            @test isempty(kernel_body_reports(GML.tensor_transpose_kernel!, (3, 2, 2),
                zeros(T, 3, 2, 2), zeros(T, 2, 3, 2)))
            @test isempty(JET.get_reports(JET.report_opt(GML.assign_ones!, (A3{T},);
                target_modules = TARGET)))
            @test isempty(kernel_body_reports(GML.assign_ones_kernel!, (3, 2),
                zeros(T, 3, 3, 2)))
        end

        # src/kernels/vec_tensor_mul.jl: `test_rrule` in test/kernels/kernel_pullbacks.jl
        @test isempty(JET.get_reports(JET.report_opt(GML.vec_tensor_mul,
            (Vector{Float64}, A3{Float64}); target_modules = TARGET)))
        @test isempty(kernel_body_reports(GML.vec_tensor_mul_kernel!, (2, 3, 2),
            zeros(2, 3, 2), zeros(2), zeros(2, 3, 2)))

        # src/kernels/mat_tensor_mul.jl
        for T in (Float32, Float64)
            @test isempty(JET.get_reports(JET.report_opt(GML.mat_tensor_mul!,
                (A3{T}, Matrix{T}, A3{T}); target_modules = TARGET)))
            @test isempty(kernel_body_reports(GML.mat_tensor_mul_kernel!, (2, 3, 2),
                zeros(T, 2, 3, 2), zeros(T, 2, 4), zeros(T, 4, 3, 2)))
            @test isempty(JET.get_reports(JET.report_opt(GML.symmetric_mat_mul!,
                (A3{T}, Vector{T}, A3{T}, Int); target_modules = TARGET)))
            @test isempty(kernel_body_reports(GML.symmetric_mat_mul_kernel!, (3, 2, 2),
                zeros(T, 3, 2, 2), zeros(T, 6), zeros(T, 3, 2, 2), 3))
        end
        @test isempty(JET.get_reports(JET.report_opt(GML.lo_mat_mul!,
            (A3{Float64}, Vector{Float64}, A3{Float64}, Int); target_modules = TARGET)))
        @test isempty(kernel_body_reports(GML.lo_mul_kernel!, (3, 2, 2), zeros(3, 2, 2),
            zeros(3), zeros(3, 2, 2), 3))
        @test isempty(JET.get_reports(JET.report_opt(GML.up_mat_mul!,
            (A3{Float64}, Vector{Float64}, A3{Float64}, Int); target_modules = TARGET)))
        @test isempty(kernel_body_reports(GML.up_mul_kernel!, (3, 2, 2), zeros(3, 2, 2),
            zeros(3), zeros(3, 2, 2), 3))
        for T in (Float16, Float32, Float64)
            @test isempty(JET.get_reports(JET.report_opt(GML.skew_mat_mul!,
                (A3{T}, Vector{T}, A3{T}, Int); target_modules = TARGET)))
            @test isempty(kernel_body_reports(GML.skew_mat_mul_kernel!, (3, 2, 2),
                zeros(T, 3, 2, 2), zeros(T, 3), zeros(T, 3, 2, 2), 3))
        end

        # src/kernels/kernel_ad_routines/: the pullbacks that launch kernels, at the `Float64`
        # types of `test_rrule` in test/kernels/kernel_pullbacks.jl and of
        # test/optimizers/structured_array_parameters.jl
        S, Ss, B, dC = zeros(3), zeros(6), zeros(3, 3, 2), zeros(3, 3, 2)
        for f in (GML.lo_mat_mul, GML.up_mat_mul, GML.skew_mat_mul)
            @test isempty(JET.get_reports(JET.report_opt(rrule(f, S, B, 3)[2],
                (A3{Float64},); target_modules = TARGET)))
        end
        @test isempty(JET.get_reports(JET.report_opt(
            rrule(GML.symmetric_mat_mul, Ss, B, 3)[2],
            (A3{Float64},); target_modules = TARGET)))
        @test isempty(JET.get_reports(JET.report_opt(
            rrule(GML.symmetric_mat_right_mul, B, Ss, 3)[2], (A3{Float64},);
            target_modules = TARGET)))
        @test isempty(kernel_body_reports(GML.lower_da_kernel!, size(dC), zero(dC), S, dC))
        @test isempty(kernel_body_reports(GML.lower_ds_kernel!, (3, 2), zeros(3, 2), B, dC))
        @test isempty(kernel_body_reports(GML.upper_da_kernel!, size(dC), zero(dC), S, dC))
        @test isempty(kernel_body_reports(GML.upper_ds_kernel!, (3, 2), zeros(3, 2), B, dC))
        @test isempty(kernel_body_reports(GML.symmetric_da_kernel!, size(dC), zero(dC), Ss,
            dC))
        @test isempty(kernel_body_reports(GML.symmetric_ds_kernel!, (6, 2), zeros(6, 2), B,
            dC))
        @test isempty(kernel_body_reports(
            GML.symmetric_right_da_kernel!, size(dC), zero(dC),
            Ss, dC))
        @test isempty(kernel_body_reports(
            GML.symmetric_right_ds_kernel!, (6, 2), zeros(6, 2),
            B, dC))
        Z, M = zeros(3, 4, 2), zeros(3, 3)
        @test isempty(JET.get_reports(JET.report_opt(
            rrule(GML.tensor_mat_skew_sym_assign, Z, M)[2], (A3{Float64},);
            target_modules = TARGET)))
        @test isempty(kernel_body_reports(GML.dz_kernel!, size(Z), zero(Z), Z, M,
            zeros(4, 4, 2), 3, 4))
        @test isempty(kernel_body_reports(GML.da_kernel!, (3, 3, 2), zeros(3, 3, 2), Z, M,
            zeros(4, 4, 2), 3, 4))
        for T in (Float32, Float64)
            @test isempty(JET.get_reports(JET.report_opt(GML.tensor_scalar_product,
                (A3{T}, A3{T}); target_modules = TARGET)))
            @test isempty(kernel_body_reports(GML.tensor_scalar_product_kernel!, 3,
                zeros(T, 3), zeros(T, 3, 4, 2), zeros(T, 3, 4, 2), 4, 2))
        end

        # src/kernels/inverses/: `cpu_inverse` and its pullback in test/kernels/tensor_inverse.jl
        @test isempty(JET.get_reports(JET.report_opt(GML.cpu_inverse, (A3{Float64},);
            target_modules = TARGET)))
        @test isempty(kernel_body_reports(GML.cpu_inverse_kernel!, 2, zero(B), B))
        Id = zeros(3, 3, 2)
        foreach(i -> Id[i, i, :] .= 1, 1:3)
        @test isempty(JET.get_reports(JET.report_opt(rrule(GML.cpu_inverse, Id)[2],
            (A3{Float64},); target_modules = TARGET)))
        @test isempty(kernel_body_reports(GML.cpu_inverse_pullback_kernel!, 2, zero(B), B,
            dC))
        for T in (Float16, Float32, Float64)
            for (f, k, n) in ((GML.tensor_inverse2!, GML.inv22_kernel!, 2),
                (GML.tensor_inverse3!, GML.inv33_kernel!, 3),
                (GML.tensor_inverse4!, GML.inv44_kernel!, 4),
                (GML.tensor_inverse5!, GML.inv55_kernel!, 5))
                @test isempty(JET.get_reports(JET.report_opt(f, (A3{T}, A3{T});
                    target_modules = TARGET)))
                @test isempty(kernel_body_reports(k, 2, zeros(T, n, n, 2), zeros(T, n, n, 2)))
            end
            @test isempty(JET.get_reports(JET.report_opt(GML.tensor_mat_skew_sym_assign!,
                (A3{T}, A3{T}, Matrix{T}); target_modules = TARGET)))
            @test isempty(kernel_body_reports(GML.tensor_mat_skew_sym_assign_kernel!,
                (4, 4, 2), zeros(T, 4, 4, 2), zeros(T, 3, 4, 2), zeros(T, 3, 3)))
        end

        # src/arrays/poisson_tensor.jl: `PoissonTensor(CPU(), 4, Float16)` in
        # test/arrays/poisson_tensor.jl; `Float32` in test/devices/metal.jl. The element type is
        # a `DataType` argument, so one line holds both.
        reports = JET.get_reports(JET.report_opt(PoissonTensor, (CPU, Int, DataType);
            target_modules = TARGET))
        @test_broken isempty(reports)  # #325
        for T in (Float16, Float32)
            @test isempty(kernel_body_reports(
                GML.assign_ones_for_poisson_tensor_kernel!, 4,
                zeros(T, 4, 4), 2))
        end

        # src/data_loader/batch.jl
        qp = (q = zeros(2, 8, 4), p = zeros(2, 8, 4))
        indices = ones(Int, 2, 4)
        TimeSeriesQP = DataLoader{
            Float64, @NamedTuple{q::A3{Float64}, p::A3{Float64}}, Nothing, :TimeSeries}
        Indices = Vector{Tuple{Int, Int}}
        # test/data_loader/draw_batch_for_tensor_test.jl
        @test isempty(JET.get_reports(JET.report_opt(
            GML.convert_input_and_batch_indices_to_array,
            (TimeSeriesQP, Batch{:Transformer}, Indices); target_modules = TARGET)))
        @test isempty(kernel_body_reports(GML.assign_input_from_vector_of_tuples_kernel!,
            (2, 3, 4), zeros(2, 3, 4), zeros(2, 3, 4), qp, indices))
        @test isempty(kernel_body_reports(GML.assign_output_from_vector_of_tuples_kernel!,
            (2, 3, 4), zeros(2, 3, 4), zeros(2, 3, 4), qp, indices, 3))
        # test/integration/docstrings/data_loader.jl
        @test isempty(JET.get_reports(JET.report_opt(
            GML.convert_input_and_batch_indices_to_array,
            (DataLoader{Float64, A3{Float64}, Nothing, :RegularData},
                Batch{:FeedForward},
                Indices); target_modules = TARGET)))
        for (T, BatchType) in ((Float32, Batch{:Transformer}), (
            Float64, Batch{:FeedForward}))
            # the optimizer functor in test/data_loader/training_history_eltype.jl
            @test isempty(JET.get_reports(JET.report_opt(
                GML.convert_input_and_batch_indices_to_array,
                (DataLoader{T, A3{T}, Nothing, :TimeSeries}, Batch{:FeedForward}, Indices);
                target_modules = TARGET)))
            # input and output: test/data_loader/data_loader_for_input_and_output.jl
            # (`Float32`, `Batch(10, 1, 1)`) and
            # test/architectures/lagrangian_neural_network_tests.jl (`Float64`, `Batch(10)`)
            reports = JET.get_reports(JET.report_opt(
                GML.convert_input_and_batch_indices_to_array,
                (DataLoader{T, A3{T}, A3{T}, :TimeSeries}, BatchType, Indices);
                target_modules = TARGET))
            @test_broken isempty(reports)  # #325
            @test isempty(kernel_body_reports(
                GML.assign_input_from_vector_of_tuples_kernel!,
                (2, 3, 4), zeros(T, 2, 3, 4), zeros(T, 2, 8, 4), indices))
            @test isempty(kernel_body_reports(
                GML.assign_output_from_vector_of_tuples_kernel!,
                (2, 3, 4), zeros(T, 2, 3, 4), zeros(T, 2, 8, 4), indices, 3))
        end
        # the `Batch` functor under `@allocated` in test/data_loader/batch_index_set.jl, and
        # the other element types of its direct calls: `Int` in
        # test/integration/docstrings/data_loader.jl, `Float64` in
        # test/data_loader/draw_batch_for_tensor_test.jl
        @test isempty(JET.get_reports(JET.report_opt(Batch(8),
            (DataLoader{Float32, A3{Float32}, Nothing, :TimeSeries},); target_modules = TARGET)))
        @test isempty(JET.get_reports(JET.report_opt(Batch(3),
            (DataLoader{Int, A3{Int}, Nothing, :TimeSeries},); target_modules = TARGET)))
        @test isempty(JET.get_reports(JET.report_opt(Batch(10, 16), (TimeSeriesQP,);
            target_modules = TARGET)))

        # src/data_loader/tensor_assign.jl: `assign_output_estimate` in
        # test/kernels/kernel_pullbacks.jl (`Float64`) and test/data_loader/accuracy.jl
        # (`Float32`); `augment_zeros` through the pullback of `assign_output_estimate`
        for T in (Float32, Float64)
            @test isempty(JET.get_reports(JET.report_opt(GML.assign_output_estimate,
                (A3{T}, Int); target_modules = TARGET)))
            @test isempty(kernel_body_reports(
                GML.assign_output_estimate_kernel!, (2, 1, 3),
                zeros(T, 2, 1, 3), zeros(T, 2, 4, 3), 4, 1))
            @test isempty(JET.get_reports(JET.report_opt(GML.augment_zeros, (A3{T}, Int);
                target_modules = TARGET)))
            @test isempty(kernel_body_reports(GML.augment_zeros_kernel!, (2, 1, 3),
                zeros(T, 2, 4, 3), zeros(T, 2, 1, 3), 4, 1))
        end
    else
        @test_skip "JET does not work on Julia $(VERSION)"  # aviatesk/JET.jl#681
    end
end
