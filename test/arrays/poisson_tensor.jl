using GeometricMachineLearning
using LinearAlgebra: I
using Test
import Random 

Random.seed!(123)

function test_setup(n2::Int, T::DataType)
    @assert iseven(n2) 
    n = n2 ÷ 2

    one_mat = Matrix{T}(I(n))
    @test PoissonTensor(n2, T) ≈ hcat(vcat(zero(one_mat), -one_mat), vcat(one_mat, zero(one_mat)))
end

function test_application(n2::Int, T::DataType)
    @assert iseven(n2) 

    𝕁 = PoissonTensor(n2, T)
    x = rand(T, n2)
    y = rand(T, n2)

    @test 𝕁(x, y) ≈ x' * (𝕁 * y) ≈ -𝕁(y, x)
end

function test_application_nt(n2::Int, T::DataType)
    @assert iseven(n2) 

    𝕁 = PoissonTensor(n2, T)
    n = n2 ÷ 2
    x = (q = rand(T, n), p = rand(T, n))
    y = (q = rand(T, n), p = rand(T, n))

    @test 𝕁(x, y) ≈ x.q' * y.p - x.p' * y.q ≈ -𝕁(y, x)
end

function test_application_to_nt(n2::Int, T::DataType)
    @assert iseven(n2)

    𝕁 = PoissonTensor(n2, T)
    n = n2 ÷ 2
    qp = (q = rand(T, n), p = rand(T, n))

    @test 𝕁 * qp ≈ (q = qp.p, p = -qp.q)
end

for n2 in 2:2:10
    for T in (Float32, Float64)
        test_setup(n2, T)
        test_application(n2, T)
        test_application_nt(n2, T)
        test_application_to_nt(n2, T)
    end
end