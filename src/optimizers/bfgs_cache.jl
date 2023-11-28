@doc raw"""
The cache for the BFGS optimizer.

It stores an array for the previous time step `B` and the inverse of the Hessian matrix `H`.

It is important to note that setting up this cache already requires a derivative! This is not the case for the other optimizers.
"""
struct BFGSCache{T, AT<:AbstractArray{T}} <: AbstractCache
    B::AT
    S::AT
    H::AbstractMatrix{T}
    function BFGSCache(B::AbstractArray)
        new{eltype(B), typeof(zero(B))}(zero(B), zero(B), initialize_hessian_inverse(zero(B)))
    end
end

@doc raw"""
In order to initialize `BGGSCache` we first need gradient information. This is why we initially have this `BFGSDummyCache` until gradient information is available.
"""
struct BFGSDummyCache{T, AT<:AbstractArray{T}} <: AbstractCache
    function BFGSDummyCache(B::AbstractArray)
        new{eltype(B), typeof(zero(B))}()
    end
end

@kernel function assign_diagonal_ones_kernel!(B::AbstractMatrix{T}) where T 
    i = @index(Global)
    B[i, i] = one(T)
end

@doc raw"""
This initializes the inverse of the Hessian for various arrays. This requires an implementation of a *vectorization operation* `vec`. This is important for custom arrays.
"""
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

setup_bfgs_dummy_cache(ps::NamedTuple) = apply_toNT(setup_bfgs_dummy_cache, ps)
setup_bfgs_dummy_cache(ps::Tuple) = Tuple([setup_bfgs_cache(x) for x in ps])
setup_bfgs_dummy_cache(B::AbstractArray) = BFGSDummyCache(B)