using GeometricMachineLearning
using GeometricMachineLearning: _custom_mul, _custom_transpose
using LinearAlgebra: norm
using Zygote: gradient

function symplectic_attention(z::NamedTuple{(:q, :p), Tuple{AT, AT}}, _ps::Union{NamedTuple, NeuralNetworkParameters}) where {AT<:AbstractArray}
    expPAP = exp.(_custom_mul(_custom_mul(_custom_transpose(z.p), _ps.L1.A), z.p))
    (q = z.q + _custom_mul(_custom_mul(_ps.L1.A, z.p), 2 * expPAP) / sum(expPAP), p = z.p)
end

function symplectic_attention_simplified(z::NamedTuple{(:q, :p), Tuple{AT, AT}}, _ps::Union{NamedTuple, NeuralNetworkParameters}) where {AT<:AbstractArray}
    (q = z.p + _custom_mul(_custom_mul(z.p, _ps.L1.A), z.p), p = z.p)
end

function symplectic_linear_map(z::NamedTuple{(:q, :p), Tuple{AT, AT}}, _ps::Union{NamedTuple, NeuralNetworkParameters}) where {AT<:AbstractArray}
    (q = z.q + _custom_mul(_ps.L1.A, z.p), p = z.p)
end

S = rand(SymmetricMatrix, 2)
ps = NeuralNetworkParameters((L1 = (A = S, ), ))
t = (q = rand(2, 2), p = rand(2, 2))

∇₁ = gradient(_ps -> norm(symplectic_attention(t, _ps)), ps)[1] # this doesn't work
∇₂ = gradient(_ps -> norm(symplectic_attention(t, _ps)), ps.params)[1] # this does work

# expPAP = expPAP = exp.(_custom_mul(_custom_mul(_custom_transpose(t.p), ps.params.L1.A), t.p))
# f₂ = (q = t.q + _custom_mul(_custom_mul(ps.params.L1.A, t.p), 2 * expPAP) / sum(expPAP), p = t.p) # this works however

l = SymplecticAttentionQ(4; symmetric = true)
nn = NeuralNetwork(Chain(l))

∇₃ = gradient(params -> norm(nn(t, params)), ps)[1] # this doesn't work

∇₄ = gradient(_ps -> norm(symplectic_linear_map(t, _ps)), ps)[1] # this works

∇₅ = gradient(_ps -> norm(symplectic_attention_simplified(t, _ps)), ps)[1]

# working means that e.g.
typeof(∇₄.params.L1.A) <: SymmetricMatrix
typeof(∇₅.params.L1.A) <: SymmetricMatrix