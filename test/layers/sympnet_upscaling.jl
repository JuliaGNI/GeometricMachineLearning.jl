using GeometricMachineLearning, Test, Zygote

function test_symplecticity(N=4, N2=20, T=Float32)
    model = Chain(PSDLayer(N, N2), GradientQ(N2, 2*N2, tanh), GradientP(N2, 2*N2, tanh), PSDLayer(N2, N))
    ps = NeuralNetwork(model, CPU(), T).params
    x = rand(T, N)
    ten = rand(T, N, N)
    # the first and last PSD layer need to have the same weight! (else they map to a different symplectic potential)
    ps[4].weight.A = ps[1].weight.A
    jacobian_matrix = Zygote.jacobian(x -> model(x, ps), x)[1]
    𝕁 = PoissonTensor(N÷2)
    @test isapprox(jacobian_matrix'*𝕁*jacobian_matrix, 𝕁, rtol=0.1)
    @test isapprox(model(ten, ps)[:,1], model(ten[:,1], ps))
end

for N=2:2:20
    for N2=2*N:2:4*N
        test_symplecticity()
    end
end