@kernel function inv22_kernel!(ˍ₋out::AT, A::AT) where {T, AT<:AbstractArray{T, 3}}
    k = @index(Global)
    begin
         @inbounds begin
                ˍ₋out[1, 1, k] = (/)((getindex)(A, 2, 2, k), (+)((*)((getindex)(A, 1, 1, k), (getindex)(A, 2, 2, k)), (*)((*)(-1, (getindex)(A, 1, 2, k)), (getindex)(A, 2, 1, k))))
                ˍ₋out[2, 1, k] = (/)((*)(-1, (getindex)(A, 2, 1, k)), (+)((*)((getindex)(A, 1, 1, k), (getindex)(A, 2, 2, k)), (*)((*)(-1, (getindex)(A, 1, 2, k)), (getindex)(A, 2, 1, k))))
                ˍ₋out[1, 2, k] = (/)((*)(-1, (getindex)(A, 1, 2, k)), (+)((*)((getindex)(A, 1, 1, k), (getindex)(A, 2, 2, k)), (*)((*)(-1, (getindex)(A, 1, 2, k)), (getindex)(A, 2, 1, k))))
                ˍ₋out[2, 2, k] = (/)((getindex)(A, 1, 1, k), (+)((*)((getindex)(A, 1, 1, k), (getindex)(A, 2, 2, k)), (*)((*)(-1, (getindex)(A, 1, 2, k)), (getindex)(A, 2, 1, k))))
                nothing
            end
    end
end

function tensor_inverse2(A::AbstractArray{T, 3}) where T 
    out = similar(A)

    tensor_inverse2!(out, A)
    
    out 
end

function tensor_inverse2!(out::AbstractArray{T, 3}, A::AbstractArray{T, 3}) where T 
    @assert size(A, 1) == size(A, 2) == 2
    @assert size(A) == size(out)

    backend = get_backend(out)
    inv22! = inv22_kernel!(backend)

    inv22!(out, A, ndrange = size(A, 3))

    nothing
end

function ChainRulesCore.rrule(::typeof(tensor_inverse2), A::AT) where {T, AT<:AbstractArray{T, 3}}
    out = tensor_inverse2(A)
    
    function tensor_inverse_pullback(out_diff::AT)

        NoTangent(), - tensor_transpose_tensor_mul(out, tensor_tensor_mul(out_diff, tensor_transpose(out)))
    end 
    out, tensor_inverse_pullback 
end 