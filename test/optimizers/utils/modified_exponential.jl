using GeometricMachineLearning: 𝔄
using GeometricMachineLearning: 𝔄exp 
using Test

# check if we recover the regular exponential function
function test(T, N, n)
    A = T(.1)*rand(T, N, n)
    B = T(.1)*rand(T, n, N)
    @test eltype(𝔄exp(A, B)) == T 
    @test isapprox(exp(A*B), 𝔄exp(A, B))
end

N_max = 10
types = (Float32, Float64)
for N = 1:N_max
    for n = 1:N
            for T in types
                test(T, N, n)
            end
    end
end