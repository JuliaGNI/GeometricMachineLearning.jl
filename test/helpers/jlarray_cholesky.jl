# A Cholesky factorization for the `JLArray` stand-in device.
#
# `JLArrays` 0.5 refuses to convert a `JLArray` to a pointer, as a device array does, so
# `LinearAlgebra.cholesky` cannot reach LAPACK on one. A real device supplies the factorization from
# its vendor library -- `Metal` through MPS, `CUDA` through cuSOLVER -- and CholeskyQR2, which
# `GeometricOptimizers` uses to draw a `StiefelManifold` and to build its `GlobalSection`, needs it.
# So the host factorizes a copy here and the factor goes back to the device: the stand-in has what
# those devices have, and every other operation still runs on the `JLArray`.
using JLArrays: JLArray
using LinearAlgebra: LinearAlgebra, Cholesky, Symmetric, cholesky

function LinearAlgebra.cholesky(A::Symmetric{T, <:JLArray{T}}; check::Bool = true) where {T}
    F = cholesky(Symmetric(Array(parent(A)), Symbol(A.uplo)); check = check)
    Cholesky(JLArray(F.factors), F.uplo, F.info)
end
