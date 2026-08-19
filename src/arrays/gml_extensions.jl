# What `GeometricMachineLearning` adds to `GeometricOptimizers`' structured matrix types.
#
# The types themselves — `SkewSymMatrix`, `SymmetricMatrix`, the triangular family, the
# Lie-algebra-horizontal lifts, `StiefelProjection` and the manifolds — are imported from
# `GeometricOptimizers` rather than redefined here; see the import block in
# `GeometricMachineLearning.jl`. This file holds the methods that are genuinely GML's, i.e. the ones
# that reach for a dependency `GeometricOptimizers` does not have.
#
# There is only one such family. `add!` is `AbstractNeuralNetworks.add!`, a *different generic
# function* from the `add!` `GeometricOptimizers` defines internally for the same types, so the
# upstream methods do not serve GML's callers. It is exported by GML and has been since 0.1.
#
# `networkbackend` needs no methods at all: `AbstractNeuralNetworks.networkbackend(::AbstractArray)`
# forwards to `KernelAbstractions.get_backend`, and `GeometricOptimizers` implements that for every
# one of these types. GML used to define both halves.

function add!(C::SkewSymMatrix, A::SkewSymMatrix, B::SkewSymMatrix)
    @assert A.n == B.n == C.n
    add!(C.S, A.S, B.S)
end

function add!(C::SymmetricMatrix, A::SymmetricMatrix, B::SymmetricMatrix)
    @assert A.n == B.n == C.n
    add!(C.S, A.S, B.S)
end

function add!(C::AT, A::AT, B::AT) where {AT <: AbstractTriangular}
    @assert A.n == B.n == C.n
    add!(C.S, A.S, B.S)
end

function add!(C::StiefelLieAlgHorMatrix, A::StiefelLieAlgHorMatrix, B::StiefelLieAlgHorMatrix)
    @assert A.N == B.N == C.N
    @assert A.n == B.n == C.n
    add!(C.A, A.A, B.A)
    add!(C.B, A.B, B.B)
end

function add!(C::GrassmannLieAlgHorMatrix, A::GrassmannLieAlgHorMatrix, B::GrassmannLieAlgHorMatrix)
    @assert A.N == B.N == C.N
    @assert A.n == B.n == C.n
    add!(C.B, A.B, B.B)
end
