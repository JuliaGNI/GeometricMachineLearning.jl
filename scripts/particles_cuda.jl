using GeometricMachineLearning
using HDF5
using LinearAlgebra
using Lux
using Printf
using Random
using Zygote
using ProgressMeter

import CUDA

fpath = "../../ReducedBasisMethods/runs/BoT_Np5e4_k_010_050_np_10_T25.h5"
file = h5open(fpath, "r")
snapshots = read(file, "snapshots")
X = snapshots["X"]
V = snapshots["V"]

#preprocess data into the currect format
function reshape_and_vcat(X, V, i, j)
    vcat(reshape(X[:,:,i,j], size(X, 2)), reshape(V[:,:,i,j], size(V, 2))) |> CUDA.cu
end

@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = tuplejoin(tuplejoin(x, y), z...)

Z = mapreduce(i -> map(j -> reshape_and_vcat(X, V, i, j), 1:size(X,4)), tuplejoin, 1:size(X,3))

_, N, time_steps, par = size(X)
n = 10

norm_fac = mapreduce(i -> norm(Z[i]), +, 1:length(Z))

#PSD  PSD_err
fpath_proj = "../../ReducedBasisMethods/runs/BoT_Np5e4_k_010_050_np_10_T25_projections.h5"
file = h5open(fpath_proj, "r")
Ψ_PSD = read(file, "Ψp")[:, 1:n] |> CUDA.cu

Ψ_PSD_full = hcat(vcat(Ψ_PSD, CUDA.zeros(N, n)), vcat(CUDA.zeros(N, n), Ψ_PSD))

PSD_err = mapreduce(i -> norm(Z[i] - Ψ_PSD_full*Ψ_PSD_full'*Z[i]), +, 1:length(Z))/norm_fac

relu(x) = max.(0, x)
Ψ_enc = Chain(Gradient(2 * N, 4 * N, relu; change_q = true),
              Gradient(2 * N, 4 * N, relu; change_q = false),
              PSDLayer(2 * N, 100; inverse = true),
              Gradient(100, 200, relu; change_q = true),
              Gradient(100, 200, relu; change_q = false),
              PSDLayer(100, 50; inverse = true),
              Gradient(50, 200, relu; change_q = true),
              Gradient(50, 200, relu; change_q = false),
              PSDLayer(50, 2 * n; inverse = true),
              Gradient(2 * n, 4 * n, relu; change_q = false),
              Gradient(2 * n, 4 * n, relu; change_q = true))
Ψ_dec = Chain(Gradient(2 * n, 4 * n, relu; change_q = false),
              Gradient(2 * n, 4 * n, relu; change_q = true),
              PSDLayer(50, 2 * n),
              Gradient(50, 200, relu; change_q = false),
              Gradient(50, 200, relu; change_q = true),
              PSDLayer(100, 50),
              Gradient(100, 200, relu; change_q = false),
              Gradient(100, 200, relu; change_q = true),
              PSDLayer(2 * N, 100),
              Gradient(2 * N, 4 * N, relu; change_q = false),
              Gradient(2 * N, 4 * N, relu; change_q = true))
reconstr = Chain(Ψ_enc, Ψ_dec)
ps_all, st_all = Lux.setup(CUDA.device(), Random.default_rng(), reconstr)

#reconstruction error for one vector
function loss(x, ps_all, st_all)
    norm(Lux.apply(reconstr, x, ps_all, st_all)[1] - x)
end

function loss_random(Z, ps_all, st_all)
    z = Z[Int(ceil(length(Z)*rand()))]
    loss(z, ps_all, st_all)
end

loss_minibatch(Z, ps_all, st_all, batch_size=10) = mapreduce(i -> loss_random(Z, ps_all, st_all), +, 1:batch_size)

loss_total(Z, ps_all, st_all) = mapreduce(i -> loss(Z[i], ps_all, st_all), +, 1:time_steps)

function loss_total(ps_all, st_all)
    loss_total = 0
    for i in 1:time_steps
        for j in 1:par
            loss_total += loss(hcat(X[:, :, i, j], V[:, :, i, j])', ps_all, st_all)
        end
    end
    loss_total / norm_fac
end

#optim = StandardOptimizer(1e-3)
optim = AdamOptimizer()
#TODO: dispatch over the optimizer
cache = init_optimizer_cache(CUDA.device(), reconstr, optim)
n_runs = Int(5e3)
err_vec = zeros(n_runs + 1)

err_vec[1] = loss_total(ps_all, st_all)
@showprogress for i in 1:n_runs
    g = Zygote.gradient(p -> loss_minibatch(p, st_all), ps_all)[1]
    #apply!(optim, nothing, reconstr, ps_all, g)
    optimization_step!(optim, reconstr, ps_all, cache, g)
    err_vec[i + 1] = loss_total(ps_all, st_all)
    #println("error is $(err_vec[i+1])")
end

@printf "PSD error: %.5e. " PSD_err
@printf "SAE error: %.5e\n" err_vec[end]

p = plot(0:n_runs, ones(n_runs+1)*PSD_err, label="PSD error", colour="red",size=(800,500))
plot!(p, 0:n_runs, err_vec, label="Training loss", linewidth=2, size=(800,500), colour=1)
png(p, "SAE_PSD_comp")

#=
function print_symplecticity(::Lux.AbstractExplicitLayer, ::NamedTuple)
end
function print_symplecticity(l::SymplecticStiefelLayer, x::NamedTuple)
    @printf "Error in symplecticity: %.5e.\n" check_symplecticity(l, x)
end
=#
#=
for i in 1:length(reconstr)
    print_symplecticity(reconstr[i], ps_all[i])
end
=#