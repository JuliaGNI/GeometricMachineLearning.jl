using Test 
using LinearAlgebra
using GeometricMachineLearning

function check_gradient(T, N::Integer, n::Integer)
    Y = rand(GrassmannManifold{T}, N, n)

    #element of the tangent space 
    Δ = rgrad(Y, randn(T, N, n))
    A = randn(T, N, n)
    V = rgrad(Y, A)
    norm(tr(Δ'*A) - metric(Y, Δ, V))
end

display(check_gradient(Float32, 10, 5))