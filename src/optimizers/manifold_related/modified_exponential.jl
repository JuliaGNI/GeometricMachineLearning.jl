@doc raw"""
    𝔄(A)

Compute ``\mathfrak{A}(A) := \sum_{n=1}^\infty \frac{1}{n!} (A)^{n-1}.``

# Implementation

This uses a Taylor expansion that iteratively adds terms with

```julia
while norm(Aⁿ) > ε
    mul!(A_temp, Aⁿ, A)
    Aⁿ .= A_temp
    rmul!(Aⁿ, inv(n))

    𝔄 += B
    n += 1 
end
```

until the norm of `Aⁿ` becomes smaller than machine precision. 
The counter `n` in the above algorithm is initialized as `2`
The matrices `Aⁿ` and `𝔄` are initialized as the identity matrix.
"""
function 𝔄(A::AbstractMatrix{T}) where T
    Aⁿ = one(A)
    C = one(A)
    A_temp = zero(A)
    n = 2
    while norm(Aⁿ) > eps(T)
        mul!(A_temp, Aⁿ, A)
        Aⁿ .= A_temp
        rmul!(Aⁿ, T(inv(n)))

        C += Aⁿ
        n += 1 
    end
    #print("\nNumber of iterations is: ", i, "\n")
    C
end

@doc raw"""
    𝔄(B̂, B̄)

Compute ``\mathfrak{A}(B', B'') := \sum_{n=1}^\infty \frac{1}{n!} ((B'')^TB')^{n-1}.``

This expression has the property ``\mathbb{I} +  B'\mathfrak{A}(B', B'')(B'')^T = \exp(B'(B'')^T).``

# Examples

```jldoctest
using GeometricMachineLearning
using GeometricMachineLearning: 𝔄
import Random
Random.seed!(123)

B = rand(StiefelLieAlgHorMatrix, 10, 2)
B̂ = hcat(vcat(.5 * B.A, B.B), vcat(one(B.A), zero(B.B)))
B̄ = hcat(vcat(one(B.A), zero(B.B)), vcat(-.5 * B.A, -B.B))

one(B̂ * B̄') + B̂ * 𝔄(B̂, B̄) * B̄' ≈ exp(B)

# output

true
```
"""
function 𝔄(B̂::AbstractMatrix{T}, B̄::AbstractMatrix{T}) where T
    𝔄(B̄' * B̂)
end

function 𝔄exp(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where T
    I + X * 𝔄(X, Y) * Y'
end