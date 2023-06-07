using Test 
using LinearAlgebra
using GeometricMachineLearning
using GeometricMachineLearning: global_section

function check_gradient(T, N::Integer, n::Integer)
    Y = rand(GrassmannManifold{T}, N, n)

    #element of the tangent space 
    Δ = rgrad(Y, randn(T, N, n))
    A = randn(T, N, n)
    V = rgrad(Y, A)
    norm(tr(Δ'*A) - metric(Y, Δ, V))
end

function global_section_test(T, N::Integer, n::Integer)
    Y = rand(GrassmannManifold{T}, N, n)
    Q = global_section(Y)
    πQ = Q[1:N, 1:n]
    norm(Y - πQ*πQ'*Y)
end

function tangent_space_rep(T, N::Integer, n::Integer)
    Y = rand(GrassmannManifold{T}, N, n)
    Δ = rgrad(Y, randn(T, N, n))
    λY = GlobalSection(Y)
    λY.λ'*Δ
end

function gloabl_tangent_space_representation(T, N::Integer, n::Integer)
    Y = rand(GrassmannManifold{T}, N, n)
    Δ = rgrad(Y, randn(T, N, n))
    λY = GlobalSection(Y)
    global_rep(λY, Δ)
end

function run_tests(T, N, n)
    display(check_gradient(T, N, n))
    display(global_section_test(T, N, n))
    display(tangent_space_rep(T, N, n))
    display(gloabl_tangent_space_representation(T, N, n))
end

run_tests(Float32, 10, 4)
