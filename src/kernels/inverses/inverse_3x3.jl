@kernel function inv33_kernel!(ˍ₋out, A)
    k = @index(Global)
    @inbounds begin
        begin
            begin
                t1 = -1
                t2 = (*)((*)(t1, (getindex)(A, 2, 3, k)), (getindex)(A, 3, 2, k))
                t3 = (*)((getindex)(A, 3, 3, k), (getindex)(A, 2, 2, k))
                t4 = (+)(t2, t3)
                t5 = (*)((*)(t1, (getindex)(A, 2, 3, k)), (getindex)(A, 3, 1, k))
                t6 = (*)((getindex)(A, 3, 3, k), (getindex)(A, 2, 1, k))
                t7 = (+)(t5, t6)
                t8 = (*)((*)(t1, t7), (getindex)(A, 1, 2, k))
                t9 = (*)((getindex)(A, 1, 1, k), t4)
                t10 = (*)((getindex)(A, 2, 1, k), (getindex)(A, 3, 2, k))
                t11 = (*)((*)(t1, (getindex)(A, 3, 1, k)), (getindex)(A, 2, 2, k))
                t12 = (+)(t10, t11)
                t13 = (*)(t12, (getindex)(A, 1, 3, k))
                t14 = (+)((+)(t8, t9), t13)
                t15 = (/)(t4, t14)
                t16 = (*)((getindex)(A, 2, 3, k), (getindex)(A, 3, 1, k))
                t17 = (*)((*)(t1, (getindex)(A, 3, 3, k)), (getindex)(A, 2, 1, k))
                t18 = (+)(t16, t17)
                t19 = (/)(t18, t14)
                t20 = (/)(t12, t14)
                t21 = (*)((getindex)(A, 3, 2, k), (getindex)(A, 1, 3, k))
                t22 = (*)((*)(t1, (getindex)(A, 1, 2, k)), (getindex)(A, 3, 3, k))
                t23 = (+)(t21, t22)
                t24 = (/)(t23, t14)
                t25 = (*)((getindex)(A, 3, 3, k), (getindex)(A, 1, 1, k))
                t26 = (*)((*)(t1, (getindex)(A, 3, 1, k)), (getindex)(A, 1, 3, k))
                t27 = (+)(t25, t26)
                t28 = (/)(t27, t14)
                t29 = (*)((getindex)(A, 1, 2, k), (getindex)(A, 3, 1, k))
                t30 = (*)((*)(t1, (getindex)(A, 1, 1, k)), (getindex)(A, 3, 2, k))
                t31 = (+)(t29, t30)
                t32 = (/)(t31, t14)
                t33 = (*)((getindex)(A, 2, 3, k), (getindex)(A, 1, 2, k))
                t34 = (*)((*)(t1, (getindex)(A, 2, 2, k)), (getindex)(A, 1, 3, k))
                t35 = (+)(t33, t34)
                t36 = (/)(t35, t14)
                t37 = (*)((getindex)(A, 2, 1, k), (getindex)(A, 1, 3, k))
                t38 = (*)((*)(t1, (getindex)(A, 2, 3, k)), (getindex)(A, 1, 1, k))
                t39 = (+)(t37, t38)
                t40 = (/)(t39, t14)
                t41 = (*)((*)(t1, (getindex)(A, 1, 2, k)), (getindex)(A, 2, 1, k))
                t42 = (*)((getindex)(A, 1, 1, k), (getindex)(A, 2, 2, k))
                t43 = (+)(t41, t42)
                t44 = (/)(t43, t14)
                @inbounds begin
                    ˍ₋out[1, 1, k] = t15
                    ˍ₋out[2, 1, k] = t19
                    ˍ₋out[3, 1, k] = t20
                    ˍ₋out[1, 2, k] = t24
                    ˍ₋out[2, 2, k] = t28
                    ˍ₋out[3, 2, k] = t32
                    ˍ₋out[1, 3, k] = t36
                    ˍ₋out[2, 3, k] = t40
                    ˍ₋out[3, 3, k] = t44
                    nothing
                end
            end
        end
    end
end

function tensor_inverse3(A::AbstractArray{T, 3}) where {T}
    out = similar(A)

    tensor_inverse3!(out, A)

    out
end

function tensor_inverse3!(out::AbstractArray{T, 3}, A::AbstractArray{T, 3}) where {T}
    @assert size(A, 1) == size(A, 2) == 3
    @assert size(A) == size(out)

    backend = networkbackend(out)
    inv33! = inv33_kernel!(backend)

    inv33!(out, A, ndrange = size(A, 3))

    nothing
end

function ChainRulesCore.rrule(::typeof(tensor_inverse3), A::AT) where {
        T, AT <: AbstractArray{T, 3}}
    out = tensor_inverse3(A)

    function tensor_inverse_pullback(out_diff)
        out_diff = unthunk(out_diff)

        NoTangent(),
        - tensor_transpose_tensor_mul(out, tensor_tensor_mul(out_diff, tensor_transpose(out)))
    end
    out, tensor_inverse_pullback
end
