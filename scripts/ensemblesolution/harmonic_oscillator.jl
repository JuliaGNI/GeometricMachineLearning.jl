using GeometricMachineLearning
using GeometricSolutions
using Test

using GeometricProblems.HarmonicOscillator
using GeometricProblems.HarmonicOscillator: reference_solution




const t₀ = 0.0
const Δt = 0.1
const nt = 10
const tspan = (t₀, Δt*nt)

const k = 0.5
const ω = √k

ϑ₁(t,q) = q[2]
ϑ₂(t,q) = zero(eltype(q))

function ϑ(q)
    p = zero(q)
    p[1] = ϑ₁(0,q)
    p[2] = ϑ₂(0,q)
    return p
end

const q₀ = [0.5, 0.0]
const p₀ = ϑ(q₀)

const A = sqrt(q₀[2]^2 / k + q₀[1]^2)
const ϕ = asin(q₀[1] / A)

const reference_solution_q = A * sin(ω * Δt * nt + ϕ)
const reference_solution_p = ω * A * cos(ω * Δt * nt + ϕ)


#creation fonction pour creer solution de reference_solution

#enregistrer pour differentes conditions initiales differentes trajectoires esapcees d'un meme pas de temps dans un tablea

#creer l'object ensemble solution

#creer le training data associé

#entrainer







