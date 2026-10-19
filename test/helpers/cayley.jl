# What the tests of the Cayley kernel `tensor_cayley` and of `VolumePreservingAttention` share: the
# bound the kernel is held to, the dense reference it is compared with, and the inputs the layer
# forms. The kernel's own rows, the device rows and the layer's forward pass all use them.
using LinearAlgebra: I, dot, opnorm

# The bound `10 n eps(T) (1 + ‖Cₖ‖₂²)` of one slice, and its largest value over a batch.
function cayley_bound(::Type{T}, Ck::AbstractMatrix, n::Int) where {T}
    10.0 * n * eps(T) * (1.0 + Float64(opnorm(Ck, 2))^2)
end

function cayley_bound(C::AbstractArray{T, 3}) where {T}
    maximum(cayley_bound(T, @view(C[:, :, k]), size(C, 1)) for k in axes(C, 3))
end

# The Cayley transform `(I + Cₖ) \ (I - Cₖ)` by a dense solve in the element type of `Cₖ`,
# independent of the kernel, of one slice and of every slice of a batch.
function cayley_reference(Ck::AbstractMatrix{T}) where {T}
    Id = Matrix{T}(I, size(Ck))
    (Id + Ck) \ (Id - Ck)
end

function cayley_reference(C::AbstractArray{T, 3}) where {T}
    stack(cayley_reference, eachslice(C; dims = 3))
end

# A batch of `k` skew `n × n` slices, `R - Rᵀ` for `R = rand(T, n, n, k)`.
function skew_batch(::Type{T}, n::Int, k::Int) where {T}
    R = rand(T, n, n, k)
    R - permutedims(R, (2, 1, 3))
end

# The `skew_sym = false` path's `C` on the host, slice by slice: the lower triangle of the scalar
# products of the columns of `x` weighted by `A`, divided by `√M`, and made skew by subtracting its
# own transpose — the same triangle, the same divisor and the same subtraction the layer makes.
function arbitrary_C(x::AbstractArray{T, 3}, A::AbstractMatrix) where {T}
    M, seq = size(x, 1), size(x, 2)
    C = zeros(T, seq, seq, size(x, 3))
    for k in axes(x, 3)
        X = Matrix(@view x[:, :, k])
        L = zeros(T, seq, seq)
        for i in 1:seq, j in 1:(i - 1)

            L[i, j] = dot(@view(X[:, i]), A * @view(X[:, j]))
        end
        C[:, :, k] = (L - L') / T(√M)
    end
    C
end
