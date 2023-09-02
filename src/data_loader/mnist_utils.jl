@kernel function assign_val_kernel!(output, target)
    i = @index(Global)
    output[target[i]+1, i] = 1 
end

function onehotbatch(target::AbstractVector{T}) where {T<:Integer}
    backend = KernelAbstractions.get_backend(target)
    output = KernelAbstractions.zeros(backend, T, 10, length(target))
    assign_val! = assign_val_kernel!(backend)
    assign_val!(output, target, ndrange=length(target))
end

# second argumen pl is "patch length"
# this splits the image into patches of size pl×pl and then arranges them into a matrix,
# the columns of the matrix give the patch number.

function flatten(image_patch::AbstractMatrix)
    n, m = size(image_patch)
    reshape(image_patch, n*m)
end

function split_and_flatten(image::AbstractMatrix, pl)
    n, m = size(image)
    @assert n == m
    @assert n%pl == 0
    #square root of patch number
    pnsq = n ÷ pl
    hcat(Tuple(vcat(map(j -> map(i -> flatten(image[pl*(i-1)+1:pl*i,pl*(j-1)+1:pl*j,1]), 1:pnsq),1:pnsq)...))...)
end

@kernel function split_and_flatten_kernel!(output::AbstractArray{T, 3}, input::AbstractArray{T, 3}, patch_length::Integer, number_of_patches::Integer) where T
    i,j,k = @index(Global, NTuple)
    patch_index = (j÷patch_length)*(Int(√number_of_patches)) + i÷patch_length + 1
    within_patch_i = i%patch_length+1
    within_patch_j = j%patch_length+1
    output[within_patch_i+(within_patch_j-1)*patch_length,patch_index,k] = input[i,j,k]
end

function split_and_flatten(input::AbstractArray{T, 3}, patch_length::Integer=7, number_of_patches::Integer=16) where T 
    backend = KernelAbstractions.get_backend(input)
    output = KernelAbstractions.allocate(backend, T, patch_length^2, number_of_patches, size(input, 3))
    split_and_flatten! = split_and_flatten_kernel!(backend)
    split_and_flatten!(output, input, patch_length, number_of_patches, ndrange=size(input))
    output 
end