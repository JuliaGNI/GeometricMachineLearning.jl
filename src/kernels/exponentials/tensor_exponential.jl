function tensor_exponential(B::AbstractArray{T, 3}) where T 
    m, m2, batch_size = size(B)
    @assert m == m2
    output = init_output(B)
    matrix_mul_tensor = copy(output)
    
    step = 0
    while true 
        step += 1
        previous_step = copy(output)

        matrix_mul_tensor = tensor_tensor_mul(matrix_mul_tensor, B)/step
        output += matrix_mul_tensor
        norm(previous_step - output)/batch_size < eps(T) ? break : nothing
    end
    output
end

function init_output(B::AbstractArray{T, 3}) where T 
    output = zero(B)
    assign_ones!(output)
    output
end

function assign_ones!(output::AbstractArray{T, 3}) where T
    backend = KernelAbstractions.get_backend(output)
    assign_ones_backend! = assign_ones_kernel!(backend)
    dims = (size(output,1), size(output,3))
    assign_ones_backend!(output, ndrange=dims)
end

@kernel function assign_ones_kernel!(output::AbstractArray{T, 3}) where T 
    i,k = @index(Global, NTuple)
    output[i,i,k] = one(T)
end

function ChainRulesCore.rrule(::typeof(init_output), B::AbstractArray{T, 3}) where T 
    output = init_output(B)
    function init_output_pullback(output_diff::Union{AbstractArray{T, 3}, Thunk}) where T 
        return NoTangent(), ZeroTangent()
    end
    output, init_output_pullback
end