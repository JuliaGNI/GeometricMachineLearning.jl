using GeometricMachineLearning: 𝔄
using GeometricMachineLearning: 𝔄exp 
using Test

# check if we recover the regular exponential function
function test(N, n)
    A = .1*rand(N, n)
    B = .1*rand(n, N)
    @test isapprox(exp(A*B), 𝔄exp(A, B))
end

N_max = 10
for N = 1:N_max
    for n = 1:N
        test(N, n)
    end
end