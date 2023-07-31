"""
Warning: all these tests seem to be fine for double precision, but require a ridicolously high tolerance (~5f-3) for single precision!
"""

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
    norm(tr(Δ'*A) - metric(Y, Δ, V))/N/n
end

function global_section_test(T, N::Integer, n::Integer)
    Y = rand(GrassmannManifold{T}, N, n)
    Q = global_section(Y)
    πQ = Q[1:N, 1:n]
    norm(Y - πQ*πQ'*Y)/N/n
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

function coordinate_chart_rep(T, N::Integer, n::Integer)
    Y = rand(GrassmannManifold{T}, N, n)
    Y.A = Y.A*inv(Y.A[1:n, 1:n])
    Y
end

function run_tests(T, N, n, tol)
    @test check_gradient(T, N, n) < tol
    @test global_section_test(T, N, n) < tol
    @test norm(tangent_space_rep(T, N, n)[1:n,1:n])/N/n < tol
    @test typeof(gloabl_tangent_space_representation(T, N, n)) <: GrassmannLieAlgHorMatrix
    # because of the matrix inversion the tolerance here is set to a higher value
    @test norm(coordinate_chart_rep(T, N, n)[1:n,1:n]-I(n))/N/n < tol*10
end

tol = 1e-10
T = Float64
for N in 1:10
    for n in 1:N 
        run_tests(T, N, n, tol)
    end
end
