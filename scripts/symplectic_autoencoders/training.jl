"""
TODO

- Implement variational autoencoder!!! Up to now the variational property has not been included.
- Make the computation of the reduction error automatic! (this has to be done for many values!)
- Also try using analytical data!!!
- At the moment you are training every network 4 times. 好傻，笨蛋。
"""

using GeometricMachineLearning 
using LinearAlgebra: svd, norm
using ProgressMeter
using Zygote
using HDF5
using CUDA
using GeometricIntegrators
using Plots

include("vector_fields.jl")
include("initial_condition.jl")

T = Float64
n_epochs = 100
n_range = 2:1:15
μ_range = (T(0.51), T(0.625), T(0.55), T(0.47))  
opt = AdamOptimizer(T.((0.001, 0.9, 0.99, 1e-8))...)
retraction = Cayley()


function gpu_backend()
    data = h5open("snapshot_matrix.h5", "r") do file
        read(file, "data")
    end
    n_params = h5open("snapshot_matrix.h5", "r") do file
        read(file, "n_params")
    end
    (CUDABackend(), data |> cu, n_params)
end 

function cpu_backend()
    data = h5open("snapshot_matrix.h5", "r") do file
        read(file, "data")
    end
    n_params = h5open("snapshot_matrix.h5", "r") do file
        read(file, "n_params")
    end
    (CPU(), data, n_params)
end 


backend, data, n_params = 
try 
    gpu_backend()
catch
    cpu_backend()
end

dl = DataLoader(data)
n_time_steps = size(data,2) / n_params
N = size(data,1)÷2


function get_psd_encoder_decoder(; n=5)
    Φ = svd(hcat(data[1:N,:], data[(N+1):2*N,:])).U[:,1:n]
    PSD = hcat(vcat(Φ, zero(Φ)), vcat(zero(Φ), Φ))

    PSD_cpu = _cpu_convert(PSD)
    psd_encoder(z) = PSD_cpu'*z 
    psd_decoder(ξ) = PSD_cpu*ξ
    psd_encoder, psd_decoder
end

function get_nn_encoder_decoder(; n=5, n_epochs=500, activation=tanh, opt=opt, T=T)
    Ψᵉ = Chain(
        GradientQ(2*N, 10*N, activation), 
        GradientP(2*N, 10*N, activation),
        GradientQ(2*N, 10*N, activation), 
        GradientP(2*N, 10*N, activation),
        PSDLayer(2*N, 2*n; retraction=retraction)
    )

    Ψᵈ = Chain(
        GradientQ(2*n, 10*n, activation), 
        GradientP(2*n, 10*n, activation),
        PSDLayer(2*n, 2*N; retraction=retraction),
        GradientQ(2*N, 2*N, activation)
         )

    model = Chain(  
                    Ψᵉ.layers..., 
                    Ψᵈ.layers...
    )

    ps = initialparameters(backend, T, model)
    loss(model, ps, dl)

    optimizer_instance = Optimizer(opt, ps)
    n_training_iterations = Int(ceil(n_epochs*dl.n_params/dl.batch_size))
    progress_object = Progress(n_training_iterations; enabled=true)

    for _ in 1:n_training_iterations
        redraw_batch!(dl)
        loss_val, pb = try 
            Zygote.pullback(ps -> loss(model, ps, dl), ps)
        catch 
            continue 
        end
        dp = pb(one(loss_val))[1]

        try 
            optimization_step!(optimizer_instance, model, ps, dp)
        catch
            continue 
        end
        ProgressMeter.next!(progress_object; showvalues=[(:TrainingLoss, loss_val)])
    end

    n_layers = length(model.layers)
    psᵉ = _cpu_convert(ps[1:length(Ψᵉ.layers)])
    psᵈ = _cpu_convert(ps[(length(Ψᵉ.layers)+1):end])   

    nn_encoder(z) = Ψᵉ(z, psᵉ)
    nn_decoder(ξ) = Ψᵈ(ξ, psᵈ)

    nn_encoder, nn_decoder
end

function get_reduced_model(encoder, decoder;n=5, μ_val=0.51, Ñ=(N-2), n_time_steps=n_time_steps, integrator=ImplicitMidpoint(), system_type=GeometricMachineLearning.Symplectic())
    params = (μ=μ_val, Ñ=Ñ, Δx=T(1/(Ñ-1)))
    timestep = T(1/(n_time_steps-1))
    timespan = (T(0), T(1))
    ics = get_initial_condition_vector(μ_val, Ñ)
    v_field_full = v_field(params)
    v_field_reduced = reduced_vector_field_from_full_explicit_vector_field(v_field_explicit(params), decoder, N, n)
    ReducedSystem(N, n, encoder, decoder, v_field_full, v_field_reduced, params, timespan, timestep, ics; integrator=integrator, system_type=system_type)
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

_cpu_convert(Y::StiefelManifold) = StiefelManifold(_cpu_convert(Y.A))

function get_reconstructed_trajectories(psd_rs, nn_rs)
    psd_time_series = perform_integration_reduced(psd_rs)
    nn_time_series = perform_integration_reduced(nn_rs)
    for t in axes(psd_time_series.q, 1)
        psd_time_series.q[t] = psd_rs.decoder(psd_time_series.q[t])
        nn_time_series.q[t] = nn_rs.decoder(nn_time_series.q[t])
    end
    (psd=psd_time_series, nn=nn_time_series, full=perform_integration_full(psd_rs))  
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

data_cpu = _cpu_convert(data)

function get_enocders_decoders(n_range)
    encoders_decoders = NamedTuple()
    for n in n_range
        psd_encoder, psd_decoder = get_psd_encoder_decoder(n=n)
        nn_encoder, nn_decoder = get_nn_encoder_decoder(n=n, n_epochs=n_epochs)
        nn_ed = (encoder=nn_encoder, decoder=nn_decoder)
        psd_ed = (encoder=psd_encoder, decoder=psd_decoder)
        encoders_decoders_current = (nn=nn_ed, psd=psd_ed)
        encoders_decoders = NamedTuple{(keys(encoders_decoders)..., Symbol("n"*string(n)))}((values(encoders_decoders)..., encoders_decoders_current))
    end
    encoders_decoders
end

encoders_decoders = get_enocders_decoders(n_range)

μ_errors = NamedTuple()
for μ_test_val in μ_range
    dummy_rs = get_reduced_model(nothing, nothing; n=1, μ_val=μ_test_val, Ñ=(N-2))
    sol_full = perform_integration_full(dummy_rs)
    errors = NamedTuple()
    for n in n_range
        current_n_identifier = Symbol("n"*string(n))

        encoders_decoders_current = encoders_decoders[current_n_identifier]
        psd_encoder, psd_decoder = encoders_decoders_current.psd
        nn_encoder, nn_decoder = encoders_decoders_current.nn

        psd_rs = get_reduced_model(psd_encoder, psd_decoder; n=n, μ_val=μ_test_val, Ñ=(N-2))
        nn_rs = get_reduced_model(nn_encoder, nn_decoder; n=n, μ_val=μ_test_val, Ñ=(N-2))

        reduction_errors = (psd=compute_reduction_error(psd_rs, sol_full), nn=compute_reduction_error(nn_rs, sol_full))
        projection_errors = (psd=compute_projection_error(psd_rs, sol_full), nn=compute_projection_error(nn_rs, sol_full))
        temp_errors = (reduction_error=reduction_errors, projection_error=projection_errors)
        errors = NamedTuple{(keys(errors)..., current_n_identifier)}((values(errors)..., temp_errors))
    end

    global μ_errors = NamedTuple{(keys(μ_errors)..., Symbol("μ"*string(μ_test_val)))}((values(μ_errors)..., errors))
end

function plot_projection_reduction_errors(μ_errors)
    number_errors = length(μ_errors[1])
    n_vals = zeros(Int, number_errors)
    nn_projection_vals = zeros(number_errors)
    nn_reduction_vals = zeros(number_errors)
    psd_projection_vals = zeros(number_errors)
    psd_reduction_vals = zeros(number_errors)
    for μ_key in keys(μ_errors)
        μ = string(μ_key)
        it = 0
        for n_key in keys(μ_errors[μ_key])
            it += 1
            n = parse(Int, string(n_key)[2:end])
            nn_projection_val = μ_errors[μ_key][n_key].projection_error.nn 
            nn_reduction_val = μ_errors[μ_key][n_key].reduction_error.nn 
            psd_projection_val = μ_errors[μ_key][n_key].projection_error.psd
            psd_reduction_val = μ_errors[μ_key][n_key].reduction_error.psd
            n_vals[it] = n 
            nn_projection_vals[it] = nn_projection_val 
            nn_reduction_vals[it] = nn_reduction_val 
            psd_projection_vals[it] = psd_projection_val
            psd_reduction_vals[it] = psd_reduction_val
        end
        plot_object = plot(n_vals, psd_projection_vals, color=2, seriestype=:scatter, markershape=:cross, ylimits=(0,1), label="PSD projection")
        plot!(plot_object, n_vals, psd_reduction_vals, color=2, seriestype=:scatter, label="PSD reduction")

        plot!(plot_object, n_vals, nn_projection_vals, color=3, seriestype=:scatter, markershape=:cross, label="NN projection")        
        plot!(plot_object, n_vals, nn_reduction_vals, color=3, seriestype=:scatter, label="NN reduction")
        png(plot_object, "plots/v3mu"*μ[3:end])
    end
end

plot_projection_reduction_errors(μ_errors)
