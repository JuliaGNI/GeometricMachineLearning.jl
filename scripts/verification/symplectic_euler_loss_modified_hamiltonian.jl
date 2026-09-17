# Check that the true Hamiltonian does not minimise SymplecticEulerLoss on exact-flow data, and
# that the residual it leaves is first order in the timestep.
#
# Run from the repo root:
#   julia --startup-file=no --project=. scripts/verification/symplectic_euler_loss_modified_hamiltonian.jl
#
# The HNN tutorial's phase space section trains on pairs carried by the exact flow of vf(z) = 𝕁z,
# and scores the network with SymplecticEulerLoss, which holds it to one symplectic Euler step.
# The two agree only to first order, so the minimiser of that loss is not H but the Hamiltonian
# whose symplectic Euler step reproduces the exact flow. This is what stops a trained loss below
# the true Hamiltonian's residual from reading as a fit better than exact. The tutorial's
# `!!! info` at docs/src/tutorials/hamiltonian_neural_network.md records the rule.
#
# The loss is reimplemented here from src/loss/symplectic_euler_loss.jl rather than called: the
# package form takes a network and its parameters, and the point of the check is the value at the
# exact Hamiltonian, which is not a network. The two must not drift apart — the expression below
# is that file's final three lines, with loss.hvf replaced by the field of H.

using LinearAlgebra: norm

const 𝕁 = [0.0 1.0; -1.0 0.0]

# X_H = (∂H/∂p, -∂H/∂q) for H = (q² + p²)/2, the Hamiltonian of vf(z) = 𝕁z.
hamiltonian_vector_field(z) = vcat(z[2:2, :], -z[1:1, :])

"""
The tutorial's grid, carried one step of `timestep` by the exact flow `exp(timestep * 𝕁)`.
"""
function exact_flow_data(timestep)
    input = hcat([[q, p] for q in -1:0.1:1 for p in -1:0.1:1]...)
    input, exp(timestep * 𝕁) * input
end

"""
`SymplecticEulerLoss` evaluated at the true Hamiltonian instead of at a network.
"""
function residual(timestep, variant)
    input, output = exact_flow_data(timestep)
    n = size(input, 1) ÷ 2
    evaluation_point = variant === :A ?
                       vcat(output[1:n, :], input[(n + 1):(2n), :]) :
                       vcat(input[1:n, :], output[(n + 1):(2n), :])
    difference_quotient = (output - input) / timestep
    norm(hamiltonian_vector_field(evaluation_point) - difference_quotient) /
    norm(difference_quotient)
end

const TIMESTEPS = (0.1, 0.05, 0.025)

for variant in (:A, :B)
    residuals = [residual(timestep, variant) for timestep in TIMESTEPS]
    ratios = residuals[1:(end - 1)] ./ residuals[2:end]

    println("variant :$(variant)")
    for (timestep, r) in zip(TIMESTEPS, residuals)
        println("  Δt = $(timestep): residual = $(r)")
    end
    println("  ratios on halving Δt: $(ratios)")

    # The true Hamiltonian leaves a residual, so it is not the minimiser.
    @assert all(>(1e-3), residuals) "the true Hamiltonian minimises the loss for variant :$(variant)"
    # First order: halving the timestep halves the residual.
    @assert all(r -> isapprox(r, 2.0; atol = 1e-2), ratios) "residual is not first order in Δt for variant :$(variant)"
end

println("\nBoth variants: the true Hamiltonian leaves a first-order residual.")
