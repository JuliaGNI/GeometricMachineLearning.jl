using GeometricMachineLearning
using Test
import Random

Random.seed!(123)

const sys_dim = 2
const model = GSympNet(sys_dim)

function set_up_and_apply_integrator(; T = Float32)
    ics₁ = (q = rand(T, sys_dim ÷ 2), p = rand(T, sys_dim ÷ 2))
    ics₂ = (q = rand(T, sys_dim ÷ 2), p = rand(T, sys_dim ÷ 2))
    𝕁 = PoissonTensor(sys_dim, T)

    product₀ = 𝕁(ics₁, ics₂)

    nn = NeuralNetwork(model, CPU(), T)

    iterates₁ = iterate(nn, ics₁)
    iterates₂ = iterate(nn, ics₂)

    final_iterate₁ = (q = iterates₁.q[:, end], p = iterates₁.p[:, end])
    final_iterate₂ = (q = iterates₂.q[:, end], p = iterates₂.p[:, end])

    product_final = 𝕁(final_iterate₁, final_iterate₂)

    @test product_final ≉ 0 ≉ product₀
end

set_up_and_apply_integrator(; T = Float32)
set_up_and_apply_integrator(; T = Float64)
