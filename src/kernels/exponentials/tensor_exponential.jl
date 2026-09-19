# The identity tensor and its `rrule`, used by `tensor_cayley.jl` and `cpu_inverse.jl`. The file
# name is wider than what it holds; renaming it, and `exponentials/` with it, is issue C18.

function init_output(B::AbstractArray{T, 3}) where {T}
    output = zero(B)
    assign_ones!(output)
    output
end

function assign_ones!(output::AbstractArray{T, 3}) where {T}
    backend = networkbackend(output)
    assign_ones_backend! = assign_ones_kernel!(backend)
    dims = (size(output, 1), size(output, 3))
    assign_ones_backend!(output, ndrange = dims)
end

@kernel function assign_ones_kernel!(output::AbstractArray{T, 3}) where {T}
    i, k = @index(Global, NTuple)
    output[i, i, k] = one(T)
end

function ChainRulesCore.rrule(::typeof(init_output), B::AbstractArray{T, 3}) where {T}
    output = init_output(B)
    function init_output_pullback(_output_diff)
        return NoTangent(), ZeroTangent()
    end
    output, init_output_pullback
end
