using OffsetArrays
include("assemble_matrix.jl")

_mul(A::OffsetMatrix, q::OffsetVector) = OffsetArray(A.parent*q.parent, OffsetArrays.Origin(0))
_mul(q₁::OffsetVector, A::OffsetMatrix, q₂::OffsetVector) = q₁.parent'*A.parent*q₂.parent
_mul(p₁::OffsetVector, p₂::OffsetVector) = p₁.parent'*p₂.parent

function v_f_hamiltonian(params)
    K = assemble_matrix(params.μ, params.Δx, params.N)
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