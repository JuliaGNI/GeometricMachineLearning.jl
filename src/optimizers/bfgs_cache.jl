@doc raw"""
    BFGSCache(B)

Make the cache for the BFGS optimizer based on the array `B`.

It stores an array for the gradient of the previous time step `B` and the inverse of the Hessian matrix `H`.

The cache for the inverse of the Hessian is initialized with the idendity.
The cache for the previous gradient information is initialized with the zero vector.

Note that the cache for `H` is changed iteratively, whereas the cache for `B` is newly assigned at every time step.
"""
struct BFGSCache{T, BT<:AbstractArray{T}, HT<:AbstractMatrix{T}} <: AbstractCache{T}
    B::BT
    H::HT
    function BFGSCache(B::AbstractArray)
        zeroB = zero(B)
        H_init = initialize_hessian_inverse(zeroB)
        new{eltype(B), typeof(zeroB), typeof(H_init)}(zero(B), H_init)
    end
end

@kernel function assign_diagonal_ones_kernel!(B::AbstractMatrix{T}) where T 
    i = @index(Global)
    B[i, i] = one(T)
end

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, C::BFGSCache)
    show(io, raw"`BFGSCache` that currently stores `B`as  ...")
    show(io, "text/plain", C.B)
    println(io, "")
    println(io, "... and `H` as")
    show(io, "text/plain", C.H)
end 

# @doc raw"""
#     initialize_hessian_inverse(B)
# 
# Initialize the inverse of the Hessian for various arrays. 
# 
# # Implementation 
# This requires an implementation of a *vectorization operation* `vec`. This is important for custom arrays.
# """
function initialize_hessian_inverse(B::AbstractArray{T}) where T
    length_of_array = length(vec(B))
    backend = KernelAbstractions.get_backend(B)
    H = KernelAbstractions.zeros(backend, T, length_of_array, length_of_array)
    assign_diagonal_ones! = assign_diagonal_ones_kernel!(backend)
    assign_diagonal_ones!(H, ndrange=length_of_array)
    H
end

setup_bfgs_cache(ps::NamedTuple) = apply_toNT(setup_bfgs_cache, ps)
setup_bfgs_cache(ps::Tuple) = Tuple([setup_bfgs_cache(x) for x in ps])
setup_bfgs_cache(B::AbstractArray) = BFGSCache(B)