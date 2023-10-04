"""
Implement variational autoencoder!!! Up to now the variational property has not been included.

Make the computation of the reduction error automatic! (this has to be done for many values!)
"""

using GeometricMachineLearning 
using LinearAlgebra: svd, norm
using ProgressMeter
using Zygote
using HDF5
using CUDA
using GeometricIntegrators

include("vector_fields.jl")
include("initial_condition.jl")

T = Float64
#μ_collection=T(5/12):T(.1):T(5/6)
n = 5
n_epochs = 200

backend, data = 
try 
    (CUDABackend(),
    h5open("snapshot_matrix.h5", "r")["data"][:,:] |> cu)
catch
    (CPU(),
    h5open("snapshot_matrix.h5", "r")["data"][:,:])
end

N = size(data,1)÷2
dl = DataLoader(data)
Φ = svd(hcat(data[1:N,:], data[(N+1):2*N,:])).U[:,1:n]
PSD = hcat(vcat(Φ, zero(Φ)), vcat(zero(Φ), Φ))

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

ps = initialparameters(backend, T, model)
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

function _cpu_convert(ps::Tuple)
    output = ()
    for elem in ps 
        output = (output..., _cpu_convert(elem))
    end
    output
end

function _cpu_convert(ps::NamedTuple)
    output = ()
    for elem in ps
        output = (output..., _cpu_convert(elem))
    end
    NamedTuple{keys(ps)}(output)
end

_cpu_convert(A::AbstractArray) = Array(A)

Ψᵉ = Chain(model.layers[1:5]...)
psᵉ = _cpu_convert(ps[1:5])
Ψᵈ = Chain(model.layers[6:end]...)
psᵈ = _cpu_convert(ps[6:end])

PSD_cpu = _cpu_convert(PSD)

nn_encoder(z) = Ψᵉ(z, psᵉ)
nn_decoder(ξ) = Ψᵈ(ξ, psᵈ)
psd_encoder(z) = PSD_cpu'*z 
psd_decoder(ξ) = PSD_cpu*ξ

data_cpu = _cpu_convert(data)
psd_error = norm(data_cpu - psd_decoder(psd_encoder(data_cpu)))/norm(data_cpu)
nn_error = norm(data_cpu - nn_decoder(nn_encoder(data_cpu)))/norm(data_cpu)

μ_test_vals = (T(0.51), T(0.625), T(0.74))

# make sure you also store tspan and tstep in the future (in the HDF5 file!!!)
function reduced_systems_for_wave_equation(μ_val, Ñ=(N-2), n=n, n_time_steps=n_time_steps; T=Float64, integrator=ImplicitMidpoint(), system_type=GeometricMachineLearning.Symplectic())
    params = (μ=μ_val, Ñ=Ñ, Δx=T(1/(Ñ-1)))
    tstep = T(1/(n_time_steps-1))
    tspan = (T(0), T(1))
    ics = get_initial_condition_vector(μ_val, Ñ)
    v_field_full = v_field(params)
    nn_v_field_reduced = reduced_vector_field_from_full_explicit_vector_field(v_field_explicit(params), nn_decoder, N, n)
    psd_v_field_reduced = reduced_vector_field_from_full_explicit_vector_field(v_field_explicit(params), psd_decoder, N, n)
    nn_rs = ReducedSystem(N, n, nn_encoder, nn_decoder, v_field_full, nn_v_field_reduced, params, tspan, tstep, ics, nn_error; integrator=integrator, system_type=system_type)
    psd_rs = ReducedSystem(N, n, psd_encoder, psd_decoder, v_field_full, psd_v_field_reduced, params, tspan, tstep, ics, psd_error; integrator=integrator, system_type=system_type)
    nn_rs, psd_rs
end

function compute_reduction_errors(μ_val=T(0.51), n_time_steps=size(data,2)/8)
    nn_rs, psd_rs = reduced_systems_for_wave_equation(μ_val, N-2, n, n_time_steps)
    (psd=compute_reduction_error(psd_rs), nn=compute_reduction_error(nn_rs))
end

function get_reconstructed_trajectories(μ_val=T(0.51), n_time_steps=size(data,2)/8)
    nn_rs, psd_rs = reduced_systems_for_wave_equation(μ_val, N-2, n, n_time_steps)
    (psd=perform_integration_reduced(psd_rs), nn=perform_integration_reduced(nn_rs), full=perform_integration_full(psd_rs))  
end

function plot_comparison_for_reconstructed_trajectories(trajectories, t_step=0)
    n_t_steps = length(trajectories.psd.t)
    t_step_index = Int(ceil(n_t_steps*t_step))
    N = length(trajectories.full.q[t_step_index])÷2
    plot_object = plot(trajectories.full.q[t_step_index][1:N], label="Numerical solution")
    plot!(plot_object, trajectories.psd.q[t_step_index][1:N], label="PSD")
    plot!(plot_object, trajectories.nn.q[t_step_index][1:N], label="NN")
    png(plot_object, "plots/comparison_for_time_step_"*string(t_step))
end