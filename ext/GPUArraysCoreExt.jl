module GPUArraysCoreExt

# `PoissonTensor`'s three array methods of `Base.:*` take a `Strided…` right-hand side. That bound
# is what keeps them from colliding with every `*(::AbstractMatrix, ::X)` that ArrayLayouts,
# FillArrays, Symbolics and GeometricOptimizers define for their own array type `X` -- seventeen
# ambiguities, and the reason for the bound is set out in `src/arrays/poisson_tensor.jl`.
#
# A GPU array is itself strided, because `CuArray` and `MtlArray` are `DenseArray`s, so the fast
# path already covers it. A *wrapped* one is not: `view(A, [1, 2, 3, 4], :)` and `A'` are a
# `SubArray` and an `Adjoint` over a GPU array, and neither is strided. Those fell through to
# Julia's generic `AbstractMatrix` multiply, which reads the tensor one element at a time, so on a
# device they raised "Scalar indexing is disallowed." instead of answering.
#
# `AnyGPUArray` is the union that covers both cases -- `AbstractGPUArray` together with
# `WrappedGPUArray` -- which is why the bound here is `AnyGPU…` and not `AbstractGPUArray`: the
# wrapped case is precisely the one that failed. It meets `AbstractMatrix` only in arrays a GPU
# package owns, so it reopens none of the seventeen: measured in a cold process with
# BandedMatrices and BlockArrays loaded, the ambiguity count is 1 with these methods and 1
# without.
#
# This file's behaviour is device-only. `test/metal/metal.jl` asserts it on an Apple GPU, which
# runs on every Apple-silicon Mac and in the `Metal` workflow; the Linux and Windows entries of the
# test matrix do not reach it.

using GPUArraysCore: AnyGPUArray, AnyGPUMatrix, AnyGPUVector
using GeometricMachineLearning
import GeometricMachineLearning: PoissonTensor, _vcat, assign_q_and_p

function Base.:*(𝕁::PoissonTensor{T}, v::AnyGPUVector{T}) where {T}
    _vcat(𝕁(assign_q_and_p(v, 𝕁.n)))
end

function Base.:*(𝕁::PoissonTensor{T}, v::AnyGPUMatrix{T}) where {T}
    _vcat(𝕁(assign_q_and_p(v, 𝕁.n)))
end

function Base.:*(𝕁::PoissonTensor{T}, v::AnyGPUArray{T, 3}) where {T}
    _vcat(𝕁(assign_q_and_p(v, 𝕁.n)))
end

end
