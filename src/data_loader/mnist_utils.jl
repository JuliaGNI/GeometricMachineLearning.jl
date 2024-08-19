@kernel function assign_val_kernel!(output, target)
    i = @index(Global)
    output[target[i]+1, i] = 1 
end

@doc raw"""
One-hot-batch encoding of a vector of integers: $input\in\{0,1,\ldots,9\}^\ell$. 
The output is a tensor of shape $10\times1\times\ell$. 
```math
0 \mapsto \begin{bmatrix} 1 & 0 & \ldots & 0 \end{bmatrix}.
```
In more abstract terms: $i \mapsto e_i$.
"""
function onehotbatch(target::AbstractVector{T}) where {T<:Integer}
    backend = KernelAbstractions.get_backend(target)
    output = KernelAbstractions.zeros(backend, T, 10, length(target))
    assign_val! = assign_val_kernel!(backend)
    assign_val!(output, target, ndrange=length(target))
    reshape(output, 10, 1, length(target))
end

# """
# Based on coordinates i,j this returns the batch index (for MNIST data set for now).
# """
function patch_index(i::T, j::T, patch_length::T, number_of_patches::T=(28÷patch_length)^2) where {T<:Integer}
    opt_i = i % patch_length == 0 ? 1 : 0
    opt_j = j % patch_length == 0 ? 1 : 0 
    (j÷(patch_length)-opt_j)*(Int(√number_of_patches)) + (i÷(patch_length)-opt_i) + 1
end

# """ 
# Based on coordinates i,j this returns the index within the batch
# """
function within_patch_index(i::T, j::T, patch_length::T) where {T<:Integer}
    opt_i = i % patch_length == 0 ? 1 : 0
    opt_j = j % patch_length == 0 ? 1 : 0 
    i_red, j_red = i%patch_length + opt_i*patch_length, j%patch_length + opt_j*patch_length
    (j_red-1)*patch_length + i_red
end

function index_conversion(i::T, j::T, patch_length::T, number_of_patches::T=(28÷patch_length)^2) where {T<:Integer}
    within_patch_index(i, j, patch_length), patch_index(i, j, patch_length, number_of_patches)
end

@kernel function split_and_flatten_kernel!(output::AbstractArray{T, 3}, input::AbstractArray{T, 3}, patch_length::Integer, number_of_patches::Integer) where T
    i,j,k = @index(Global, NTuple)
    patch_index₁, patch_index₂ = index_conversion(i, j, patch_length, number_of_patches)
    output[patch_index₁, patch_index₂, k] = input[i, j, k]
end

@doc raw"""
    split_and_flatten(input::AbstractArray)::AbstractArray

Perform a preprocessing of an image into *flattened patches*.

This rearranges the input data (in an intricate way) so that it can easily be processed with a transformer.

# Examples

Consider a matrix of size ``6\times6`` which we want to divide into patches of size ``3\times3``.

```jldoctest
using GeometricMachineLearning

input = [ 1  2  3  4  5  6; 
          7  8  9 10 11 12; 
         13 14 15 16 17 18;
         19 20 21 22 23 24; 
         25 26 27 28 29 30; 
         31 32 33 34 35 36]

split_and_flatten(input; patch_length = 3, number_of_patches = 4)

# output

9×4 Matrix{Int64}:
  1  19   4  22
  7  25  10  28
 13  31  16  34
  2  20   5  23
  8  26  11  29
 14  32  17  35
  3  21   6  24
  9  27  12  30
 15  33  18  36
```

Here we see that `split_and_flatten`:
1. *splits* the original matrix into four ``3\times3`` matrices and then 
2. *flattens* each matrix into a column vector of size ``9.``
After this all the vectors are put together again to yield a ``9\times4`` matrix.

# Arguments

The optional keyword arguments are: 
- `patch_length`: by default this is 7. 
- `number_of_patches`: by default this is 16.

The sizes of the first and second axis of the output of `split_and_flatten` are 
1. ``\mathtt{path\_length}^2`` and 
2. `number_of_patches`.
"""
function split_and_flatten(input::AbstractArray{T, 3}; patch_length::Integer=7, number_of_patches::Integer=16) where T
    @assert size(input, 1) * size(input, 2) == (patch_length ^ 2) * number_of_patches
    backend = KernelAbstractions.get_backend(input)
    output = KernelAbstractions.allocate(backend, T, patch_length^2, number_of_patches, size(input, 3))
    split_and_flatten! = split_and_flatten_kernel!(backend)
    split_and_flatten!(output, input, patch_length, number_of_patches, ndrange=size(input))
    output 
end

function split_and_flatten(input::AbstractMatrix; kwargs...)
    output = split_and_flatten(reshape(input, size(input)..., 1); kwargs...)
    reshape(output, size(output)[1:2]...)
end