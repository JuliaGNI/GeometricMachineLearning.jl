using GeometricMachineLearning
using GeometricMachineLearning: _custom_mul, _custom_transpose
using LinearAlgebra: norm
using Zygote: gradient

function single_multiplication(a::AT, _ps::Union{NamedTuple, NeuralNetworkParameters}) where {AT<:AbstractArray}
    _custom_mul(a, _ps.L1.A)
end

function double_multiplication(a::AT, _ps::Union{NamedTuple, NeuralNetworkParameters}) where {AT<:AbstractArray}
    _custom_mul(_ps.L1.A, _custom_mul(a, _ps.L1.A))
end

S = rand(SymmetricMatrix, 4)
ps = NeuralNetworkParameters((L1 = (A = S, ), ))
t = rand(4, 4)

∇₁ = gradient(_ps -> norm(double_multiplication(t, _ps)), ps)[1] # this doesn't work
∇₂ = gradient(_ps -> norm(double_multiplication(t, _ps)), ps.params)[1] # this does work

∇₃ = gradient(_ps -> norm(single_multiplication(t, _ps)), ps)[1] # this does work
∇₄ = gradient(_ps -> norm(single_multiplication(t, _ps)), ps.params)[1] # this does work

# working means that e.g.
(typeof(∇₁.params.L1.A) <: SymmetricMatrix) |> display
(typeof(∇₂.L1.A) <: SymmetricMatrix) |> display
(typeof(∇₃.params.L1.A) <: SymmetricMatrix) |> display
(typeof(∇₄.L1.A) <: SymmetricMatrix) |> display