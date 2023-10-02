"""
Implement variational autoencoder!!! Up to now the variational property has not been included.
"""

using GeometricMachineLearning 
using LinearAlgebra: svd, norm
using ProgressMeter
using Zygote
using HDF5

include("vector_fields.jl")

T = Float64
#μ_collection=T(5/12):T(.1):T(5/6)
n = 5
n_epochs = 2000
backend = CPU()

data = h5open("snapshot_matrix.h5", "r")["data"]
data = reshape(data, size(data,1), size(data,2)*size(data,3))
N = size(data,1)÷2
dl = DataLoader(data)
Φ = svd(hcat(data[1:N,:], data[(N+1):2*N,:])).U[:,1:n]
PSD = hcat(vcat(Φ, zero(Φ)), vcat(zero(Φ), Φ))
PSD_error = norm(data - PSD*PSD'*data)/norm(data)

activation = tanh
model = Chain(  GradientQ(2*N, 2*N, activation), 
                GradientP(2*N, 2*N, activation),
                PSDLayer(2*N, 2*n),
                GradientQ(2*n, 2*n, activation),
                GradientP(2*n, 2*n, activation),
                GradientQ(2*n, 2*n, activation),
                GradientP(2*n, 2*n, activation),
                PSDLayer(2*n, 2*N),
                GradientQ(2*N, 2*N, activation),
                GradientP(2*N, 2*N, activation)
)

ps = initialparameters(backend, Float32, model)
loss(model, ps, dl)

optimizer_instance = Optimizer(AdamOptimizer(), ps)
n_training_iterations = Int(ceil(n_epochs*dl.n_params/dl.batch_size))
progress_object = Progress(n_training_iterations; enabled=true)

for _ in 1:n_training_iterations
    redraw_batch!(dl)
    loss_val, pb = Zygote.pullback(ps -> loss(model, ps, dl), ps)
    dp = pb(one(loss_val))[1]

    optimization_step!(optimizer_instance, model, ps, dp)
    ProgressMeter.next!(progress_object; showvalues=[(:TrainingLoss, loss_val)])
end

Ψᵈ = Chain(model.layers[6:end])
psᵈ = ps[6:end]
μ_test_vals = (T(0.51), T(0.625), T(0.74))

function build_reduced_vector_field(μ_val, N=N)
    params = (μ=μ_val, N=N, Δx=T(1/(N-1)))
    K = assemble_matrix(params.μ, params.Δx, params.N)
    full_mat = hcat(vcat(K + K', zero(K)), vcat(zero(K), one(K)))
    𝕁n = SymplecticPotential(n)
    function v_reduced(v, t, z, params)
        v .= 𝕁n * Zygote.jacobian(z -> Ψᵈ(z,psᵈ), z)[1]' * full_mat * Ψᵈ(z, psᵈ)
    end
    v_reduced
end

function perform_integration_reduced(μ_val, n_time_steps, N=N)
    tspan = (T(0),T(1))
    tstep = T((tspan[2] - tspan[1])/(n_time_steps-1))
    ics_offset = get_initial_condition(μ_val, N+2)
    ics = vcat(ics_offset.q.parent, ics_offset.p.parent)
    params = (μ=μ_val, N=N, Δx=T(1/(N-1)))
    ode = ODEProblem(build_reduced_vector_field(μ_val, N), parameters=params, tspan, tstep, ics)
    integrate(ode, ImplicitMidpoint())
end

function compute_reduction_error()
    sol₁ = perform_integration_reduced(μ_val, n_time_steps, N)
    sol₂ = perform_integration(params, n_time_steps)
    sol_matrix₁ = zeros(2*sys_dim, n_time_steps)
        ...
    sol_matrix₂ = zeros(2*sys_dim, n_time_steps)
    for (t_ind,q,p) in zip(1:n_time_steps,sol.q,sol.p)
        sols_matrix[:, n_time_steps*μ_ind+t_ind] = vcat(q,p)
    end
    norm(sol_matrix₁ - sol_matrix₂)
end
