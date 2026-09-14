@kernel function inv44_kernel!(ˍ₋out, A)
    k = @index(Global)
    @inbounds begin
        begin
            begin
                t1 = (*)((getindex)(A, 4, 4, k), (getindex)(A, 3, 3, k))
                t2 = -1
                t3 = (*)((*)(t2, (getindex)(A, 4, 3, k)), (getindex)(A, 3, 4, k))
                t4 = (+)(t1, t3)
                t5 = (*)((getindex)(A, 2, 2, k), t4)
                t6 = (*)((getindex)(A, 4, 4, k), (getindex)(A, 3, 2, k))
                t7 = (*)((*)(t2, (getindex)(A, 4, 2, k)), (getindex)(A, 3, 4, k))
                t8 = (+)(t6, t7)
                t9 = (*)((*)(t2, t8), (getindex)(A, 2, 3, k))
                t10 = (*)((*)(t2, (getindex)(A, 4, 2, k)), (getindex)(A, 3, 3, k))
                t11 = (*)((getindex)(A, 4, 3, k), (getindex)(A, 3, 2, k))
                t12 = (+)(t10, t11)
                t13 = (*)(t12, (getindex)(A, 2, 4, k))
                t14 = (+)((+)(t5, t9), t13)
                t15 = (*)((getindex)(A, 2, 1, k), t4)
                t16 = (*)((*)(t2, (getindex)(A, 4, 1, k)), (getindex)(A, 3, 3, k))
                t17 = (*)((getindex)(A, 4, 3, k), (getindex)(A, 3, 1, k))
                t18 = (+)(t16, t17)
                t19 = (*)(t18, (getindex)(A, 2, 4, k))
                t20 = (*)((*)(t2, (getindex)(A, 4, 1, k)), (getindex)(A, 3, 4, k))
                t21 = (*)((getindex)(A, 3, 1, k), (getindex)(A, 4, 4, k))
                t22 = (+)(t20, t21)
                t23 = (*)((*)(t2, t22), (getindex)(A, 2, 3, k))
                t24 = (+)((+)(t15, t19), t23)
                t25 = (*)((*)(t2, (getindex)(A, 1, 2, k)), t24)
                t26 = (*)(t14, (getindex)(A, 1, 1, k))
                t27 = (*)((*)(t2, (getindex)(A, 4, 1, k)), (getindex)(A, 3, 2, k))
                t28 = (*)((getindex)(A, 4, 2, k), (getindex)(A, 3, 1, k))
                t29 = (+)(t27, t28)
                t30 = (*)((getindex)(A, 2, 3, k), t29)
                t31 = (*)((getindex)(A, 2, 1, k), t12)
                t32 = (*)((*)(t2, (getindex)(A, 2, 2, k)), t18)
                t33 = (+)((+)(t30, t31), t32)
                t34 = (*)((*)(t2, (getindex)(A, 1, 4, k)), t33)
                t35 = (*)((*)(t2, (getindex)(A, 2, 2, k)), t22)
                t36 = (*)((getindex)(A, 2, 1, k), t8)
                t37 = (*)((getindex)(A, 2, 4, k), t29)
                t38 = (+)((+)(t35, t36), t37)
                t39 = (*)(t38, (getindex)(A, 1, 3, k))
                t40 = (+)((+)((+)(t25, t26), t34), t39)
                t41 = (/)(t14, t40)
                t42 = (*)((*)(t2, (getindex)(A, 2, 1, k)), t4)
                t43 = (*)((*)(t2, t18), (getindex)(A, 2, 4, k))
                t44 = (*)(t22, (getindex)(A, 2, 3, k))
                t45 = (+)((+)(t42, t43), t44)
                t46 = (/)(t45, t40)
                t47 = (/)(t38, t40)
                t48 = (*)((*)(t2, (getindex)(A, 2, 3, k)), t29)
                t49 = (*)((*)(t2, (getindex)(A, 2, 1, k)), t12)
                t50 = (*)((getindex)(A, 2, 2, k), t18)
                t51 = (+)((+)(t48, t49), t50)
                t52 = (/)(t51, t40)
                t53 = (*)(t8, (getindex)(A, 1, 3, k))
                t54 = (*)((*)(t2, (getindex)(A, 1, 2, k)), t4)
                t55 = (*)((*)(t2, (getindex)(A, 1, 4, k)), t12)
                t56 = (+)((+)(t53, t54), t55)
                t57 = (/)(t56, t40)
                t58 = (*)(t18, (getindex)(A, 1, 4, k))
                t59 = (*)(t4, (getindex)(A, 1, 1, k))
                t60 = (*)((*)(t2, t22), (getindex)(A, 1, 3, k))
                t61 = (+)((+)(t58, t59), t60)
                t62 = (/)(t61, t40)
                t63 = (*)((*)(t2, t8), (getindex)(A, 1, 1, k))
                t64 = (*)((*)(t2, (getindex)(A, 1, 4, k)), t29)
                t65 = (*)((getindex)(A, 1, 2, k), t22)
                t66 = (+)((+)(t63, t64), t65)
                t67 = (/)(t66, t40)
                t68 = (*)((*)(t2, (getindex)(A, 1, 2, k)), t18)
                t69 = (*)(t29, (getindex)(A, 1, 3, k))
                t70 = (*)(t12, (getindex)(A, 1, 1, k))
                t71 = (+)((+)(t68, t69), t70)
                t72 = (/)(t71, t40)
                t73 = (*)((*)(t2, (getindex)(A, 4, 3, k)), (getindex)(A, 2, 4, k))
                t74 = (*)((getindex)(A, 4, 4, k), (getindex)(A, 2, 3, k))
                t75 = (+)(t73, t74)
                t76 = (*)((getindex)(A, 1, 2, k), t75)
                t77 = (*)((*)(t2, (getindex)(A, 4, 2, k)), (getindex)(A, 2, 4, k))
                t78 = (*)((getindex)(A, 2, 2, k), (getindex)(A, 4, 4, k))
                t79 = (+)(t77, t78)
                t80 = (*)((*)(t2, t79), (getindex)(A, 1, 3, k))
                t81 = (*)((*)(t2, (getindex)(A, 4, 2, k)), (getindex)(A, 2, 3, k))
                t82 = (*)((getindex)(A, 4, 3, k), (getindex)(A, 2, 2, k))
                t83 = (+)(t81, t82)
                t84 = (*)((getindex)(A, 1, 4, k), t83)
                t85 = (+)((+)(t76, t80), t84)
                t86 = (/)(t85, t40)
                t87 = (*)((getindex)(A, 2, 1, k), (getindex)(A, 4, 4, k))
                t88 = (*)((*)(t2, (getindex)(A, 4, 1, k)), (getindex)(A, 2, 4, k))
                t89 = (+)(t87, t88)
                t90 = (*)(t89, (getindex)(A, 1, 3, k))
                t91 = (*)((getindex)(A, 4, 3, k), (getindex)(A, 2, 1, k))
                t92 = (*)((*)(t2, (getindex)(A, 4, 1, k)), (getindex)(A, 2, 3, k))
                t93 = (+)(t91, t92)
                t94 = (*)((*)(t2, (getindex)(A, 1, 4, k)), t93)
                t95 = (*)((*)(t2, t75), (getindex)(A, 1, 1, k))
                t96 = (+)((+)(t90, t94), t95)
                t97 = (/)(t96, t40)
                t98 = (*)(t79, (getindex)(A, 1, 1, k))
                t99 = (*)((*)(t2, t89), (getindex)(A, 1, 2, k))
                t100 = (*)((*)(t2, (getindex)(A, 4, 1, k)), (getindex)(A, 2, 2, k))
                t101 = (*)((getindex)(A, 2, 1, k), (getindex)(A, 4, 2, k))
                t102 = (+)(t100, t101)
                t103 = (*)(t102, (getindex)(A, 1, 4, k))
                t104 = (+)((+)(t98, t99), t103)
                t105 = (/)(t104, t40)
                t106 = (*)((getindex)(A, 1, 2, k), t93)
                t107 = (*)((*)(t2, t102), (getindex)(A, 1, 3, k))
                t108 = (*)((*)(t2, t83), (getindex)(A, 1, 1, k))
                t109 = (+)((+)(t106, t107), t108)
                t110 = (/)(t109, t40)
                t111 = (*)((getindex)(A, 3, 4, k), (getindex)(A, 2, 2, k))
                t112 = (*)((*)(t2, (getindex)(A, 2, 4, k)), (getindex)(A, 3, 2, k))
                t113 = (+)(t111, t112)
                t114 = (*)(t113, (getindex)(A, 1, 3, k))
                t115 = (*)((*)(t2, (getindex)(A, 2, 3, k)), (getindex)(A, 3, 2, k))
                t116 = (*)((getindex)(A, 2, 2, k), (getindex)(A, 3, 3, k))
                t117 = (+)(t115, t116)
                t118 = (*)((*)(t2, t117), (getindex)(A, 1, 4, k))
                t119 = (*)((*)(t2, (getindex)(A, 2, 4, k)), (getindex)(A, 3, 3, k))
                t120 = (*)((getindex)(A, 3, 4, k), (getindex)(A, 2, 3, k))
                t121 = (+)(t119, t120)
                t122 = (*)((*)(t2, (getindex)(A, 1, 2, k)), t121)
                t123 = (+)((+)(t114, t118), t122)
                t124 = (/)(t123, t40)
                t125 = (*)((getindex)(A, 2, 1, k), (getindex)(A, 3, 4, k))
                t126 = (*)((*)(t2, (getindex)(A, 3, 1, k)), (getindex)(A, 2, 4, k))
                t127 = (+)(t125, t126)
                t128 = (*)((*)(t2, t127), (getindex)(A, 1, 3, k))
                t129 = (*)(t121, (getindex)(A, 1, 1, k))
                t130 = (*)((getindex)(A, 2, 1, k), (getindex)(A, 3, 3, k))
                t131 = (*)((*)(t2, (getindex)(A, 3, 1, k)), (getindex)(A, 2, 3, k))
                t132 = (+)(t130, t131)
                t133 = (*)((getindex)(A, 1, 4, k), t132)
                t134 = (+)((+)(t128, t129), t133)
                t135 = (/)(t134, t40)
                t136 = (*)((getindex)(A, 1, 2, k), t127)
                t137 = (*)((getindex)(A, 2, 1, k), (getindex)(A, 3, 2, k))
                t138 = (*)((*)(t2, (getindex)(A, 3, 1, k)), (getindex)(A, 2, 2, k))
                t139 = (+)(t137, t138)
                t140 = (*)((*)(t2, (getindex)(A, 1, 4, k)), t139)
                t141 = (*)((*)(t2, t113), (getindex)(A, 1, 1, k))
                t142 = (+)((+)(t136, t140), t141)
                t143 = (/)(t142, t40)
                t144 = (*)(t139, (getindex)(A, 1, 3, k))
                t145 = (*)((*)(t2, (getindex)(A, 1, 2, k)), t132)
                t146 = (*)(t117, (getindex)(A, 1, 1, k))
                t147 = (+)((+)(t144, t145), t146)
                t148 = (/)(t147, t40)
                @inbounds begin
                    ˍ₋out[1, 1, k] = t41
                    ˍ₋out[2, 1, k] = t46
                    ˍ₋out[3, 1, k] = t47
                    ˍ₋out[4, 1, k] = t52
                    ˍ₋out[1, 2, k] = t57
                    ˍ₋out[2, 2, k] = t62
                    ˍ₋out[3, 2, k] = t67
                    ˍ₋out[4, 2, k] = t72
                    ˍ₋out[1, 3, k] = t86
                    ˍ₋out[2, 3, k] = t97
                    ˍ₋out[3, 3, k] = t105
                    ˍ₋out[4, 3, k] = t110
                    ˍ₋out[1, 4, k] = t124
                    ˍ₋out[2, 4, k] = t135
                    ˍ₋out[3, 4, k] = t143
                    ˍ₋out[4, 4, k] = t148
                    nothing
                end
            end
        end
    end
end

function tensor_inverse4(A::AbstractArray{T, 3}) where {T}
    out = similar(A)

    tensor_inverse4!(out, A)

    out
end

function tensor_inverse4!(out::AbstractArray{T, 3}, A::AbstractArray{T, 3}) where {T}
    @assert size(A, 1) == size(A, 2) == 4
    @assert size(A) == size(out)

    backend = networkbackend(out)
    inv44! = inv44_kernel!(backend)

    inv44!(out, A, ndrange = size(A, 3))

    nothing
end

function ChainRulesCore.rrule(::typeof(tensor_inverse4), A::AT) where {
        T, AT <: AbstractArray{T, 3}}
    out = tensor_inverse4(A)

    function tensor_inverse_pullback(out_diff)
        out_diff = unthunk(out_diff)

        NoTangent(),
        - tensor_transpose_tensor_mul(out, tensor_tensor_mul(out_diff, tensor_transpose(out)))
    end
    out, tensor_inverse_pullback
end
