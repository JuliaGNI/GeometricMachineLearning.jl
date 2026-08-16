# Elementwise arithmetic bridges between GML's structured matrix types and GeometricOptimizers.
#
# GeometricOptimizers builds its optimizer caches out of a handful of in-place primitives -- `_add!`,
# `_rac!` (elementwise square root), `_square!`, `_div!` and `_rmul!`. Its generic methods broadcast
# over `AbstractArray`, which does not work for GML's structured matrices: they store only their free
# parameters in a vector `S` (or `A`/`B`) and either have no `setindex!` at all or would silently
# symmetrise what is written to them.
#
# For every one of these types the free parameters *are* the coordinates the optimizer should work
# in, so each bridge is the corresponding operation on the storage. GO dispatches on its own
# `SkewSymMatrix`/`StiefelLieAlgHorMatrix`, which are distinct types from GML's, hence the need to
# define these here rather than relying on GO's own methods.
#
# `GeometricOptimizers.update_section!` is bridged for the same reason: its Euclidean method is
# `Λᵗ.Y .= Λ⁽ᵗ⁻¹⁾.Y .+ B⁽ᵗ⁻¹⁾`, a broadcast into the parameter.

# --- SkewSymMatrix ---------------------------------------------------------------------------

GeometricOptimizers._add!(a::SkewSymMatrix{T}, b::SkewSymMatrix{T}) where T = (a.S .+= b.S; a)
GeometricOptimizers._add!(a::SkewSymMatrix{T}, b::T) where T = (a.S .+= b; a)
GeometricOptimizers._rac!(B::SkewSymMatrix, A::SkewSymMatrix) = (B.S .= sqrt.(A.S); B)
GeometricOptimizers._square!(B::SkewSymMatrix, A::SkewSymMatrix) = (B.S .= A.S .^ 2; B)
function GeometricOptimizers._div!(C::SkewSymMatrix, A::SkewSymMatrix, B::SkewSymMatrix)
    C.S .= A.S ./ B.S
    C
end

# --- StiefelLieAlgHorMatrix ------------------------------------------------------------------

function GeometricOptimizers._add!(A::StiefelLieAlgHorMatrix{T},
        B::StiefelLieAlgHorMatrix{T}) where T
    GeometricOptimizers._add!(A.A, B.A)
    A.B .+= B.B
    A
end
function GeometricOptimizers._add!(A::StiefelLieAlgHorMatrix{T}, b::T) where T
    GeometricOptimizers._add!(A.A, b)
    A.B .+= b
    A
end
function GeometricOptimizers._rac!(B::StiefelLieAlgHorMatrix, A::StiefelLieAlgHorMatrix)
    GeometricOptimizers._rac!(B.A, A.A)
    B.B .= sqrt.(A.B)
    B
end
function GeometricOptimizers._square!(B::StiefelLieAlgHorMatrix, A::StiefelLieAlgHorMatrix)
    GeometricOptimizers._square!(B.A, A.A)
    B.B .= A.B .^ 2
    B
end
function GeometricOptimizers._div!(C::StiefelLieAlgHorMatrix, A::StiefelLieAlgHorMatrix,
        B::StiefelLieAlgHorMatrix)
    GeometricOptimizers._div!(C.A, A.A, B.A)
    C.B .= A.B ./ B.B
    C
end

# --- GrassmannLieAlgHorMatrix ----------------------------------------------------------------

function GeometricOptimizers._add!(A::GrassmannLieAlgHorMatrix, B::GrassmannLieAlgHorMatrix)
    A.B .+= B.B
    A
end
GeometricOptimizers._add!(A::GrassmannLieAlgHorMatrix, b::Number) = (A.B .+= b; A)
function GeometricOptimizers._rac!(B::GrassmannLieAlgHorMatrix, A::GrassmannLieAlgHorMatrix)
    B.B .= sqrt.(A.B)
    B
end
function GeometricOptimizers._square!(B::GrassmannLieAlgHorMatrix, A::GrassmannLieAlgHorMatrix)
    B.B .= A.B .^ 2
    B
end
function GeometricOptimizers._div!(C::GrassmannLieAlgHorMatrix, A::GrassmannLieAlgHorMatrix,
        B::GrassmannLieAlgHorMatrix)
    C.B .= A.B ./ B.B
    C
end

# --- Euclidean parameters with structured storage ---------------------------------------------
#
# These are ordinary vector-space parameters (`SymmetricMatrix` in the SympNet and symplectic
# attention layers, `SkewSymMatrix` in volume-preserving attention, `LowerTriangular` and
# `UpperTriangular` in the volume-preserving feedforward layers), so the optimizer treats them like
# any other array -- it just cannot broadcast into them.

for MT in (:SymmetricMatrix, :LowerTriangular, :UpperTriangular)
    @eval begin
        GeometricOptimizers._add!(a::$MT{T}, b::$MT{T}) where T = (a.S .+= b.S; a)
        GeometricOptimizers._add!(a::$MT{T}, b::T) where T = (a.S .+= b; a)
        GeometricOptimizers._rac!(B::$MT, A::$MT) = (B.S .= sqrt.(A.S); B)
        GeometricOptimizers._square!(B::$MT, A::$MT) = (B.S .= A.S .^ 2; B)
        GeometricOptimizers._div!(C::$MT, A::$MT, B::$MT) = (C.S .= A.S ./ B.S; C)
    end
end

for MT in (:SymmetricMatrix, :SkewSymMatrix, :LowerTriangular, :UpperTriangular)
    @eval begin
        GeometricOptimizers._rmul!(a::$MT, b) = (rmul!(a.S, b); a)

        function GeometricOptimizers.update_section!(
                Λᵗ::GeometricOptimizers.GlobalSection{T, <:$MT{T}, Nothing},
                Λ⁽ᵗ⁻¹⁾::GeometricOptimizers.GlobalSection{T, <:$MT{T}, Nothing},
                B⁽ᵗ⁻¹⁾::$MT{T}, retraction) where T
            Λᵗ.Y.S .= Λ⁽ᵗ⁻¹⁾.Y.S .+ B⁽ᵗ⁻¹⁾.S
            Λᵗ
        end
    end
end
