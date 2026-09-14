@kernel function inv22_kernel!(ˍ₋out, A)
    k = @index(Global)
    @inbounds begin
        begin
            begin
                t1 = -1
                t2 = (*)((*)(t1, (getindex)(A, 1, 2, k)), (getindex)(A, 2, 1, k))
                t3 = (*)((getindex)(A, 1, 1, k), (getindex)(A, 2, 2, k))
                t4 = (+)(t2, t3)
                t5 = (/)((getindex)(A, 2, 2, k), t4)
                t6 = (*)(t1, (getindex)(A, 2, 1, k))
                t7 = (/)(t6, t4)
                t8 = (*)(t1, (getindex)(A, 1, 2, k))
                t9 = (/)(t8, t4)
                t10 = (/)((getindex)(A, 1, 1, k), t4)
                @inbounds begin
                    ˍ₋out[1, 1, k] = t5
                    ˍ₋out[2, 1, k] = t7
                    ˍ₋out[1, 2, k] = t9
                    ˍ₋out[2, 2, k] = t10
                    nothing
                end
            end
        end
    end
end

function tensor_inverse2(A::AbstractArray{T, 3}) where {T}
    out = similar(A)

    tensor_inverse2!(out, A)

    out
end

function tensor_inverse2!(out::AbstractArray{T, 3}, A::AbstractArray{T, 3}) where {T}
    @assert size(A, 1) == size(A, 2) == 2
    @assert size(A) == size(out)

    backend = networkbackend(out)
    inv22! = inv22_kernel!(backend)

    inv22!(out, A, ndrange = size(A, 3))

    nothing
end

function ChainRulesCore.rrule(::typeof(tensor_inverse2), A::AT) where {
        T, AT <: AbstractArray{T, 3}}
    out = tensor_inverse2(A)

    function tensor_inverse_pullback(out_diff)
        out_diff = unthunk(out_diff)

        NoTangent(),
        - tensor_transpose_tensor_mul(out, tensor_tensor_mul(out_diff, tensor_transpose(out)))
    end
    out, tensor_inverse_pullback
end
