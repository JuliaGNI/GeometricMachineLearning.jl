using GeometricMachineLearning: tensor_inverse4, tensor_inverse6

function test66_inverse(k::Int = 10)
    A = rand(6, 6, k)

    A_inv = tensor_inverse6(A)
    for i = 1:k
        @test inv(A[:, :, i]) ≈ A_inv[:, :, i]
    end
end

test66_inverse()

function test66_inverse_pullback(k::Int = 10)
    A = rand(6, 6, k)

    pullback_total = Zygote.pullback(tensor_inverse6, A)

    out_diff = rand(6, 6, k)

    for i = 1:k 
        pullback_k = Zygote.pullback(inv, A[:, :, k])
        
        @test pullback_total[1][:, :, k] ≈ pullback_k[1]
        @test pullback_total[2](out_diff)[1][:, :, k] ≈ pullback_k[2](out_diff[:, :, k])[1]
    end
end

test66_inverse_pullback()

function test55_inverse(k::Int = 10)
    A = rand(5, 5, k)

    A_inv = tensor_inverse5(A)
    for i = 1:k
        @test inv(A[:, :, i]) ≈ A_inv[:, :, i]
    end
end

test55_inverse()

function test66_inverse_pullback(k::Int = 10)
    A = rand(5, 5, k)

    pullback_total = Zygote.pullback(tensor_inverse5, A)

    out_diff = rand(5, 5, k)

    for i = 1:k 
        pullback_k = Zygote.pullback(inv, A[:, :, k])
        
        @test pullback_total[1][:, :, k] ≈ pullback_k[1]
        @test pullback_total[2](out_diff)[1][:, :, k] ≈ pullback_k[2](out_diff[:, :, k])[1]
    end
end

test55_inverse_pullback()

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