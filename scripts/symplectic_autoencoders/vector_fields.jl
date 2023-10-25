using OffsetArrays
include("assemble_matrix.jl")

_mul(A::OffsetMatrix, q::OffsetVector) = OffsetArray(A.parent*q.parent, OffsetArrays.Origin(0))
_mul(q₁::OffsetVector, A::OffsetMatrix, q₂::OffsetVector) = q₁.parent'*A.parent*q₂.parent
_mul(p₁::OffsetVector, p₂::OffsetVector) = p₁.parent'*p₂.parent

function v_f_hamiltonian(params)
    K = assemble_matrix(params.μ, params.Δx, params.Ñ)
    function f(f, t, q, p, params)
        #f .= -(_mul(K + OffsetArray(K.parent', OffsetArrays.Origin(0)), q))
        f .= - (K.parent + K.parent') * q / params.Δx
    end
    function v(v, t, q, p, params)
        v .= params.Δx * p / params.Δx
    end
    function hamiltonian(t, q, p, params)
        q'*K.parent*q + eltype(q)(.5) * params.Δx * p'*p
    end
    (v, f, hamiltonian)
end

function v_field(params)
    K = assemble_matrix(params.μ, params.Δx, params.Ñ).parent 
    full_mat = hcat(vcat(K + K', zero(K)), vcat(zero(K), one(K)*params.Δx))
    𝕁N = SymplecticPotential(size(K, 1))
    function v(v, t, q, params)
        v .= 𝕁N*full_mat * q / params.Δx 
    end
    v 
end

function v_field_explicit(params)
    K = assemble_matrix(params.μ, params.Δx, params.Ñ).parent 
    full_mat = hcat(vcat(K + K', zero(K)), vcat(zero(K), one(K)*params.Δx))
    𝕁N = SymplecticPotential(size(K, 1))
    function v(t, q, params)
        𝕁N*full_mat * q / params.Δx 
    end
    v 
end