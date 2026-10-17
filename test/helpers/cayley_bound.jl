# The bound of the Cayley kernel's orthogonality and value rows, `10 n eps(T) (1 + ‖Cₖ‖₂²)`, which
# `tensor_cayley`'s docstring states. One definition, because the kernel's own rows, the device rows
# and the layer's forward pass all compare against it and must stay in step with that docstring.
using LinearAlgebra: opnorm

function cayley_bound(::Type{T}, Ck::AbstractMatrix, n::Int) where {T}
    10.0 * n * eps(T) * (1.0 + Float64(opnorm(Ck, 2))^2)
end
