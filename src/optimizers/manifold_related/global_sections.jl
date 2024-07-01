@doc raw"""
    GlobalSection(Y::AbstractMatrix)

Construct a global section for `Y`.  

A global section ``\lambda`` is a mapping from a homogeneous space ``\mathcal{M}`` to the corresponding Lie group ``G`` such that 

```math 
\lambda(Y)E = Y,
```

Also see [`apply_section`](@ref) and [`global_rep`](@ref).

# Implementation

For an implementation of `GlobalSection` for a custom array (especially manifolds), the function [`global_section`](@ref) has to be generalized.
"""
struct GlobalSection{T, AT, λT} 
    Y::AT
    # for now the only lift that is implemented is the Stiefel one - these types will have to be expanded!
    λ::λT

    function GlobalSection(Y::AbstractVecOrMat)
        λ = global_section(Y)
       new{eltype(Y), typeof(Y), typeof(λ)}(Y, λ) 
    end
end

function GlobalSection(ps::NamedTuple)
    apply_toNT(GlobalSection, ps)
end

function GlobalSection(ps::Tuple)
    [GlobalSection(ps_elem) for ps_elem in ps] |> Tuple
end

@doc raw"""
    Matrix(λY::GlobalSection)

Put `λY` into matrix form. 

This is not recommended if speed is important!

Use [`apply_section`](@ref) and [`global_rep`](@ref) instead!
"""
function Base.Matrix(λY::GlobalSection)
    hcat(Matrix(λY.Y), Matrix(λY.λ))
end

@doc raw"""
    *(λY, Y)

Apply the element `λY` onto `Y`.

Here `λY` is an element of a Lie group and `Y` is an element of a homogeneous space.
"""
Base.:*(λY::GlobalSection, Y::Manifold) = apply_section(λY, Y)

@doc raw"""
    apply_section(λY::GlobalSection{T, AT}, Y₂::AT) where {T, AT <: StiefelManifold{T}}

Apply `λY` to `Y₂`.

Mathematically this is the group action of the element ``\lambda{}Y\in{}G`` on the element ``Y_2`` of the homogeneous space ``\mathcal{M}``.

Internally it calls the inplace version [`apply_section!`](@ref).
"""
function apply_section(λY::GlobalSection{T, AT}, Y₂::AT) where {T, AT<:StiefelManifold{T}}
    Y = StiefelManifold(zero(Y₂.A))
    apply_section!(Y, λY, Y₂)

    Y
end

@doc raw"""
    apply_section!(Y::AT, λY::GlobalSection{T, AT}, Y₂::AT) where {T, AT<:StiefelManifold{T}}

Apply `λY` to `Y₂` and store the result in `Y`.

The inplace version of [`apply_section`](@ref).
"""
function apply_section!(Y::AT, λY::GlobalSection{T, AT}, Y₂::AT) where {T, AT<:StiefelManifold{T}}
    N, n = size(λY.Y)

    @views Y.A .= λY.Y * Y₂.A[1:n, :] + λY.λ * Y₂.A[(n+1):N, :]
end

function apply_section(λY::GlobalSection{T, AT}, Y₂::AT) where {T, AT<:GrassmannManifold{T}}
    Y = GrassmannManifold(zero(Y₂.A))
    apply_section!(Y, λY, Y₂)

    Y
end

function apply_section!(Y::AT, λY::GlobalSection{T, AT}, Y₂::AT) where {T, AT<:GrassmannManifold{T}}
    N, n = size(λY.Y)

    @views Y.A = λY.Y * Y₂.A[1:n, :] + λY.λ * Y₂.A[(n + 1):N, :]
end

function apply_section(λY::GlobalSection{T}, Y₂::AbstractVecOrMat{T}) where {T}
    Y = copy(Y₂)
    apply_section!(Y, λY, Y₂)
end

function apply_section!(Y::AT, λY::GlobalSection{T, AT}, Y₂::AbstractVecOrMat{T}) where {T, AT<:AbstractVecOrMat{T}}
    Y .= Y₂ + λY.Y
end

function apply_section(λY::NamedTuple, Y₂::NamedTuple)
    apply_toNT(apply_section, λY, Y₂)
end

function apply_section!(Y::NamedTuple, λY::NamedTuple, Y₂::NamedTuple)
    apply_toNT(apply_section!, Y, λY, Y₂)
end

function global_rep(λY::NamedTuple, gx::NamedTuple)
    apply_toNT(global_rep, λY, gx)
end

##auxiliary function 
function global_rep(::GlobalSection{T}, gx::AbstractVecOrMat{T}) where {T}
    gx
end

@doc raw"""
    global_rep(λY::GlobalSection{T, AT}, Δ::AbstractMatrix{T}) where {T, AT<:StiefelManifold{T}}

Express `Δ` (an the tangent space of `Y`) as an instance of `StiefelLieAlgHorMatrix`.

This maps an element from ``T_Y\mathcal{M}`` to an element of ``\mathfrak{g}^\mathrm{hor}``. 

These two spaces are isomorphic where the isomorphism where the isomorphism is established through ``\lambda(Y)\in{}G`` via:

```math 
T_Y\mathcal{M} \to \mathfrak{g}^{\mathrm{hor}}, \Delta \mapsto \lambda(Y)^{-1}\Omega(Y, \Delta)\lambda(Y).
```

Also see [`GeometricMachineLearning.Ω`](@ref).

# Examples

```jldoctest
using GeometricMachineLearning
using GeometricMachineLearning: _round
import Random 

Random.seed!(123)

Y = rand(StiefelManifold, 6, 3)
Δ = rgrad(Y, randn(6, 3))
λY = GlobalSection(Y)

_round(global_rep(λY, Δ); digits = 3)

# output

6×6 StiefelLieAlgHorMatrix{Float64, SkewSymMatrix{Float64, Vector{Float64}}, Matrix{Float64}}:
  0.0     0.679   1.925   0.981  -2.058   0.4
 -0.679   0.0     0.298  -0.424   0.733  -0.919
 -1.925  -0.298   0.0    -1.815   1.409   1.085
 -0.981   0.424   1.815   0.0     0.0     0.0
  2.058  -0.733  -1.409   0.0     0.0     0.0
 -0.4     0.919  -1.085   0.0     0.0     0.0
```

# Implementation

The function `global_rep` does in fact not perform the entire map ``\lambda(Y)^{-1}\Omega(Y, \Delta)\lambda(Y)`` but only

```math
\Delta \mapsto \mathrm{skew}(Y^T\Delta),
```

to get the small skew-symmetric matrix and 

```math
\Delta \mapsto (\lambda(Y)_{[1:N, n:N]}^T \Delta)_{[1:(N-n), 1:n]},
```

for the arbitrary matrix.
"""
function global_rep(λY::GlobalSection{T, AT}, Δ::AbstractMatrix{T}) where {T, AT<:StiefelManifold{T}}
    N, n = size(λY.Y)
    StiefelLieAlgHorMatrix(
        SkewSymMatrix(λY.Y.A' * Δ),
        λY.λ' * Δ, 
        N, 
        n
    )
end

@doc raw"""
    global_rep(λY::GlobalSection{T, AT}, Δ::AbstractMatrix{T}) where {T, AT<:GrassmannManifold{T}}

Express `Δ` (an the tangent space of `Y`) as an instance of `GrassmannLieAlgHorMatrix`.

The method `global_rep` for [`GrassmannManifold`](@ref) is similar to that for [`StiefelManifold`](@ref).

# Examples

```jldoctest
using GeometricMachineLearning
using GeometricMachineLearning: _round
import Random 

Random.seed!(123)

Y = rand(GrassmannManifold, 6, 3)
Δ = rgrad(Y, randn(6, 3))
λY = GlobalSection(Y)

_round(global_rep(λY, Δ); digits = 3)

# output

6×6 GrassmannLieAlgHorMatrix{Float64, Matrix{Float64}}:
  0.0     0.0     0.0     0.981  -2.058   0.4
  0.0     0.0     0.0    -0.424   0.733  -0.919
  0.0     0.0     0.0    -1.815   1.409   1.085
 -0.981   0.424   1.815   0.0     0.0     0.0
  2.058  -0.733  -1.409   0.0     0.0     0.0
 -0.4     0.919  -1.085   0.0     0.0     0.0
```
"""
function global_rep(λY::GlobalSection{T, AT}, Δ::AbstractMatrix{T}) where {T, AT<:GrassmannManifold{T}}
    N, n = size(λY.Y)
    GrassmannLieAlgHorMatrix(
        λY.λ' * Δ,
        N,
        n
    )
end

function update_section!(Λ⁽ᵗ⁻¹⁾::GlobalSection{T, MT}, B⁽ᵗ⁻¹⁾::AbstractLieAlgHorMatrix{T}, retraction) where {T, MT <: Manifold}
    N, n = B⁽ᵗ⁻¹⁾.N, B⁽ᵗ⁻¹⁾.n
    expB = retraction(B⁽ᵗ⁻¹⁾)
    apply_section!(expB, Λ⁽ᵗ⁻¹⁾, expB)
    Λ⁽ᵗ⁻¹⁾.Y.A .= @view expB.A[:, 1:n]
    Λ⁽ᵗ⁻¹⁾.λ .= @view expB.A[:, (n+1):N]

    nothing
end

function update_section!(Λ⁽ᵗ⁻¹⁾::GlobalSection{T, AT}, B⁽ᵗ⁻¹⁾::AT, retraction) where {T, AT <: AbstractVecOrMat{T}}
    expB = retraction(B⁽ᵗ⁻¹⁾)
    apply_section!(expB, Λ⁽ᵗ⁻¹⁾, expB)
    Λ⁽ᵗ⁻¹⁾.Y .= expB

    nothing
end

function update_section!(Λ⁽ᵗ⁻¹⁾::NamedTuple, B⁽ᵗ⁻¹⁾::NamedTuple, retraction)
    update_section_closure!(Λ⁽ᵗ⁻¹⁾, B⁽ᵗ⁻¹⁾) = update_section!(Λ⁽ᵗ⁻¹⁾, B⁽ᵗ⁻¹⁾, retraction)
    apply_toNT(update_section_closure!, Λ⁽ᵗ⁻¹⁾, B⁽ᵗ⁻¹⁾)

    nothing
end