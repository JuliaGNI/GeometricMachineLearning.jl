@kernel function assign_val_kernel!(output, target)
    i = @index(Global)
    output[target[i]+1, i] = 1 
end

function onehotbatch(target::AbstractVector{T}) where {T<:Integer}
    backend = KernelAbstractions.get_backend(target)
    output = KernelAbstractions.zeros(backend, T, 10, length(target))
    assign_val! = assign_val_kernel!(backend)
    assign_val!(output, target, ndrange=length(target))
    reshape(output, 10, 1, length(target))
end

"""
Based on coordinates i,j this returns the batch index (for MNIST data set for now).
"""
function patch_index(i::T, j::T, patch_length::T, number_of_patches::T) where {T<:Integer}
    opt_i = i % patch_length == 0 ? 1 : 0
    opt_j = j % patch_length == 0 ? 1 : 0 
    (j÷(patch_length)-opt_j)*(Int(√number_of_patches)) + (i÷(patch_length)-opt_i) + 1
end

""" 
Based on coordinates i,j this returns the index within the batch
"""
function within_patch_index(i::T, j::T, patch_length::T) where {T<:Integer}
    opt_i = i % patch_length == 0 ? 1 : 0
    opt_j = j % patch_length == 0 ? 1 : 0 
    i_red, j_red = i%1 + opt_i, j%patch_length + opt_j
    j_red*patch_length + i_red
end

@kernel function split_and_flatten_kernel!(output::AbstractArray{T, 3}, input::AbstractArray{T, 3}, patch_length::Integer, number_of_patches::Integer) where T
    i,j,k = @index(Global, NTuple)
    patch_index₁ = patch_index(i, j, patch_length, number_of_patches)
    patch_index₂ = within_patch_index(i, j, patch_length)
    output[patch_index₂, patch_index₁, k] = input[i, j, k]
end

function split_and_flatten(input::AbstractArray{T, 3}, patch_length::Integer=7, number_of_patches::Integer=16) where T 
    backend = KernelAbstractions.get_backend(input)
    output = KernelAbstractions.allocate(backend, T, patch_length^2, number_of_patches, size(input, 3))
    split_and_flatten! = split_and_flatten_kernel!(backend)
    split_and_flatten!(output, input, patch_length, number_of_patches, ndrange=size(input))
    output 
end