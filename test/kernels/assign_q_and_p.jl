using GeometricMachineLearning: assign_q_and_p
using Zygote: gradient
using Test
import Random

Random.seed!(123)

rows(x, r) = x[r, ntuple(_ -> :, ndims(x) - 1)...]

# A loss that reads one half of the split sees the gradient of that half and zeros on the other, in
# the shape of the input.
@testset "assign_q_and_p: the gradient of one half" begin
    N = 3
    for T in (Float32, Float64), x in (rand(T, 2N), rand(T, 2N, 4), rand(T, 2N, 4, 5))

        g = gradient(z -> sum(assign_q_and_p(z, N).p), x)[1]
        @test g isa typeof(x)
        @test g == vcat(zeros(T, size(rows(x, 1:N))), ones(T, size(rows(x, 1:N))))
        g = gradient(z -> sum(assign_q_and_p(z, N).q), x)[1]
        @test g == vcat(ones(T, size(rows(x, 1:N))), zeros(T, size(rows(x, 1:N))))
    end
end

@testset "assign_q_and_p: the halves are the first N rows and the N after them" begin
    N = 2
    for T in (Float32, Float64),
        x in (rand(T, 2N + 1), rand(T, 2N + 1, 3), rand(T, 2N + 1, 3, 4))

        for n in (N, Int32(N))
            qp = assign_q_and_p(x, n)
            @test qp.q == rows(x, 1:N)
            @test qp.p == rows(x, (N + 1):(2N))
            @test qp.p isa typeof(x)
        end
    end
end
