using GeometricMachineLearning, LinearAlgebra, KernelAbstractions, Test, Zygote

function svd_test(N, n, train_steps=1000, tol=1e-10)
    A = rand(N, N)
    U, Σ, Vt = svd(A)
    U_result = U[:, 1:n]

    model = Chain(StiefelLayer(N, n), StiefelLayer(n, N))
    ps = initialparameters(CPU(), Float64, model)
    error(ps) = norm(A - model(A, ps))

    o₁ = Optimizer(GradientOptimizer(), ps)
    o₂ = Optimizer(MomentumOptimizer(), ps)
    o₃ = Optimizer(AdamOptimizer(), ps)

    U₁ = train_network!(o₁, model, ps, A, train_steps, tol)
    U₂ = train_network!(o₂, model, ps, A, train_steps, tol)
    U₃ = train_network!(o₃, model, ps, A, train_steps, tol)

    println("best: ", norm(U_result*U_result'*A - A))

    @test check(U₁) < tol
    @test norm(U₁ - U_result) < tol
    @test check(U₂) < tol 
    @test norm(U₂ - U_result) < tol 
    @test check(U₃) < tol 
    @test norm(U₃ - U_result) < tol 
end

function train_network!(o::Optimizer, model::Chain, ps::Tuple, A::AbstractMatrix, train_steps, tol)
    error(ps) = norm(A - model(A, ps))

    for _ in 1:train_steps
        dx = Zygote.gradient(error, ps)[1]
        optimization_step!(o, model, ps, dx)
        #println(error(ps))
    end
    ps[1].weight
end

svd_test(10, 5)