using GeometricMachineLearning: tensor_inverse2, tensor_inverse3, tensor_inverse4, tensor_inverse5, cpu_inverse
using Test
import Zygote

function test55_inverse(k::Int = 10)
    A = rand(5, 5, k)

    A_inv = tensor_inverse5(A)
    for i = 1:k
        @test inv(A[:, :, i]) ≈ A_inv[:, :, i]
    end
end

# test55_inverse()

function test55_inverse_pullback(k::Int = 10)
    A = rand(5, 5, k)

    pullback_total = Zygote.pullback(tensor_inverse5, A)

    out_diff = rand(5, 5, k)

    for i = 1:k 
        pullback_k = Zygote.pullback(inv, A[:, :, k])
        
        @test pullback_total[1][:, :, k] ≈ pullback_k[1]
        @test pullback_total[2](out_diff)[1][:, :, k] ≈ pullback_k[2](out_diff[:, :, k])[1]
    end
end

# test55_inverse_pullback()

function test44_inverse(k::Int = 10)
    A = rand(4, 4, k)

    A_inv = tensor_inverse4(A)
    for i = 1:k
        @test inv(A[:, :, i]) ≈ A_inv[:, :, i]
    end
end

test44_inverse()

function test44_inverse_pullback(k::Int = 10)
    A = rand(4, 4, k)

    pullback_total = Zygote.pullback(tensor_inverse4, A)

    out_diff = rand(4, 4, k)

    for i = 1:k 
        pullback_k = Zygote.pullback(inv, A[:, :, k])
        
        @test pullback_total[1][:, :, k] ≈ pullback_k[1]
        @test pullback_total[2](out_diff)[1][:, :, k] ≈ pullback_k[2](out_diff[:, :, k])[1]
    end
end

test44_inverse_pullback()

function test33_inverse(k::Int = 10)
    A = rand(3, 3, k)

    A_inv = tensor_inverse3(A)
    for i = 1:k
        @test inv(A[:, :, i]) ≈ A_inv[:, :, i]
    end
end

test33_inverse()


function test33_inverse_pullback(k::Int = 10)
    A = rand(3, 3, k)

    pullback_total = Zygote.pullback(tensor_inverse3, A)

    out_diff = rand(3, 3, k)

    for i = 1:k 
        pullback_k = Zygote.pullback(inv, A[:, :, k])
        
        @test pullback_total[1][:, :, k] ≈ pullback_k[1]
        @test pullback_total[2](out_diff)[1][:, :, k] ≈ pullback_k[2](out_diff[:, :, k])[1]
    end
end

test33_inverse_pullback()

function test22_inverse(k::Int = 10)
    A = rand(2, 2, k)

    A_inv = tensor_inverse2(A)
    for i = 1:k
        @test inv(A[:, :, i]) ≈ A_inv[:, :, i]
    end
end

test22_inverse()

function test22_inverse_pullback(k::Int = 10)
    A = rand(2, 2, k)

    pullback_total = Zygote.pullback(tensor_inverse2, A)

    out_diff = rand(2, 2, k)

    for i = 1:k 
        pullback_k = Zygote.pullback(inv, A[:, :, k])
        
        @test pullback_total[1][:, :, k] ≈ pullback_k[1]
        @test pullback_total[2](out_diff)[1][:, :, k] ≈ pullback_k[2](out_diff[:, :, k])[1]
    end
end

test22_inverse_pullback()

function test_cpu_inverse_pullback(k::Int = 10)
    A = rand(3, 3, k)

    pullback_total = Zygote.pullback(cpu_inverse, A)

    out_diff = rand(3, 3, k)

    for i = 1:k 
        pullback_k = Zygote.pullback(inv, A[:, :, k])
        
        @test pullback_total[1][:, :, k] ≈ pullback_k[1]
        @test pullback_total[2](out_diff)[1][:, :, k] ≈ pullback_k[2](out_diff[:, :, k])[1]
    end
end

test_cpu_inverse_pullback()