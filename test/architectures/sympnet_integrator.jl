using GeometricMachineLearning
using Test
import Random

Random.seed!(123)

const sys_dim = 2
const model = GSympNet(sys_dim)

function set_up_and_apply_integrator(; T = Float32)
    ics₁ = (q = rand(T, sys_dim ÷ 2), p = rand(T, sys_dim ÷ 2))
    ics₂ = (q = rand(T, sys_dim ÷ 2), p = rand(T, sys_dim ÷ 2))

    # The symplectic form of the two points, `𝕁(v₁, v₂) = v₁.q' * v₂.p - v₁.p' * v₂.q`.
    product₀ = ics₁.q' * ics₂.p - ics₁.p' * ics₂.q

    nn = NeuralNetwork(model, CPU(), T)

    iterates₁ = integrate(nn, ics₁)
    iterates₂ = integrate(nn, ics₂)

    final_iterate₁ = (q = iterates₁.q[:, end], p = iterates₁.p[:, end])
    final_iterate₂ = (q = iterates₂.q[:, end], p = iterates₂.p[:, end])

    product_final = final_iterate₁.q' * final_iterate₂.p -
                    final_iterate₁.p' * final_iterate₂.q

    @test product_final ≉ 0 ≉ product₀
end

set_up_and_apply_integrator(; T = Float32)
set_up_and_apply_integrator(; T = Float64)
