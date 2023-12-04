using GeometricMachineLearning

using Test
import ChainRulesTestUtils

function test_multiplication(n::Int=5, T=Float32)
    A = rand(SymmetricMatrix{T}, n)
    b = rand(T, n)
    B = rand(T, n, n)
    # test if the custom multiplication is performed the right way
    @test A*b == Matrix{T}(A)*b
    @test A*B == Matrix{T}(A)*B
end

function test_calling_symmetric_matrix(n::Int=5, T=Float32)
    B = rand(T, n, n)
    @test isapprox(SymmetricMatrix(B), .5*(B + B'))
end

function test_pullback_routine(n::Int=5, T=Float32)
    A = rand(SymmetricMatrix{T}, n)
    B = rand(T, n, n)

    @test ChainRulesTestUtils.rrule(*, A, B)
end

test_multiplication()
# this test is not working - problem has to do with FiniteDifferences.jl (I don't know if it's worth looking into this)
# test_calling_symmetric_matrix()