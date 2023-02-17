
using GeometricMachineLearning
using HDF5
using LinearAlgebra
using Lux
using Printf
using Random
using Zygote


fpath = "../ReducedBasisMethods/runs/BoT_Np5e4_k_010_050_np_10_T25.h5"
file = h5open(fpath, "r")
snapshots = read(file, "snapshots")
X = snapshots["X"]
V = snapshots["V"]

_, n, time_steps, par = size(X)
m = 10

norm_fac = 0
for i in 1:time_steps
    for j in 1:par
        global norm_fac += norm(hcat(X[:, :, i, j], V[:, :, i, j])')
    end
end

#PSD  PSD_err
fpath_proj = "../ReducedBasisMethods/runs/BoT_Np5e4_k_010_050_np_10_T25_projections.h5"
file = h5open(fpath_proj, "r")
Ψ_PSD = read(file, "Ψp")[:, 1:m]
PSD_err = 0
for i in 1:time_steps
    for j in 1:par
        global PSD_err += norm(hcat(X[:, :, i, j] * Ψ_PSD * Ψ_PSD',
                                    V[:, :, i, j] * Ψ_PSD * Ψ_PSD')' -
                               hcat(X[:, :, i, j], V[:, :, i, j])')
    end
end

PSD_err = PSD_err / norm_fac

#=
relu(x) = max.(0, x)
Ψ_enc = Chain(Gradient(2 * n, 4 * n, relu; change_q = true, init_scale = Lux.ones32),
              Gradient(2 * n, 4 * n, relu; change_q = false, init_scale = Lux.ones32),
              SymplecticStiefelLayer(2 * m, 2 * n; inverse = true),
              Gradient(2 * m, 4 * m, relu; change_q = false, init_scale = Lux.ones32),
              Gradient(2 * m, 4 * m, relu; change_q = true, init_scale = Lux.ones32))
Ψ_dec = Chain(Gradient(2 * m, 4 * m, relu; change_q = false, init_scale = Lux.ones32),
              Gradient(2 * m, 4 * m, relu; change_q = true, init_scale = Lux.ones32),
              SymplecticStiefelLayer(2 * m, 2 * n),
              Gradient(2 * n, 4 * n, relu; change_q = true, init_scale = Lux.ones32),
              Gradient(2 * n, 4 * n, relu; change_q = false, init_scale = Lux.ones32))

reconstr = Chain(Ψ_enc, Ψ_dec)
ps_all, st_all = Lux.setup(Random.default_rng(), reconstr)
ps_all[3].weight .= hcat(vcat(Ψ_PSD, zeros(n, m)), vcat(zeros(n, m), Ψ_PSD))
ps_all[8].weight .= hcat(vcat(Ψ_PSD, zeros(n, m)), vcat(zeros(n, m), Ψ_PSD))
=#
relu(x) = max.(0, x)
Ψ_enc = Chain(Gradient(2 * n, 4 * n, relu; change_q = true),
              Gradient(2 * n, 4 * n, relu; change_q = false),
              Gradient(2 * n, 4 * n, relu; change_q = true),
              Gradient(2 * n, 4 * n, relu; change_q = false),
              SymplecticStiefelLayer(100, 2 * n; inverse = true),
              Gradient(100, 200, relu; change_q = true),
              Gradient(100, 200, relu; change_q = false),
              Gradient(100, 200, relu; change_q = true),
              Gradient(100, 200, relu; change_q = false),
              SymplecticStiefelLayer(50, 100; inverse = true),
              Gradient(50, 200, relu; change_q = true),
              Gradient(50, 200, relu; change_q = false),
              Gradient(50, 200, relu; change_q = true),
              Gradient(50, 200, relu; change_q = false),
              SymplecticStiefelLayer(2 * m, 50; inverse = true),
              Gradient(2 * m, 4 * m, relu; change_q = false),
              Gradient(2 * m, 4 * m, relu; change_q = true),
              Gradient(2 * m, 4 * m, relu; change_q = false),
              Gradient(2 * m, 4 * m, relu; change_q = true))
Ψ_dec = Chain(Gradient(2 * m, 4 * m, relu; change_q = false),
              Gradient(2 * m, 4 * m, relu; change_q = true),
              Gradient(2 * m, 4 * m, relu; change_q = false),
              Gradient(2 * m, 4 * m, relu; change_q = true),
              SymplecticStiefelLayer(2 * m, 50),
              Gradient(50, 200, relu; change_q = false),
              Gradient(50, 200, relu; change_q = true),
              Gradient(50, 200, relu; change_q = false),
              Gradient(50, 200, relu; change_q = true),
              SymplecticStiefelLayer(50, 100),
              Gradient(100, 200, relu; change_q = false),
              Gradient(100, 200, relu; change_q = true),
              Gradient(100, 200, relu; change_q = false),
              Gradient(100, 200, relu; change_q = true),
              SymplecticStiefelLayer(100, 2 * n),
              Gradient(2 * n, 4 * n, relu; change_q = false),
              Gradient(2 * n, 4 * n, relu; change_q = true),
              Gradient(2 * n, 4 * n, relu; change_q = false),
              Gradient(2 * n, 4 * n, relu; change_q = true))
reconstr = Chain(Ψ_enc, Ψ_dec)
ps_all, st_all = Lux.setup(Random.default_rng(), reconstr)

function loss(x, ps_all, st_all)
    norm(Lux.apply(reconstr, x, ps_all, st_all)[1] - x)
end

function loss_minibatch(ps_all, st_all, batch_size = 10)
    loss_total = 0
    for _ in 1:batch_size
        time_step = Int(ceil(time_steps * rand()))
        par_number = Int(ceil(par * rand()))
        loss_total += loss(hcat(X[:, :, time_step, par_number],
                                V[:, :, time_steps, par_number])', ps_all, st_all)
    end
    loss_total / batch_size
end

function loss_total(ps_all, st_all)
    loss_total = 0
    for i in 1:time_steps
        for j in 1:par
            loss_total += loss(hcat(X[:, :, i, j], V[:, :, i, j])', ps_all, st_all)
        end
    end
    loss_total / norm_fac
end

optim = MomentumOptimizer(1e-3, 5e-1)
g = gradient(p -> loss_minibatch(p, st_all), ps_all)[1]
state = MomentumOptimizerCache(optim, reconstr, ps_all, g)
n_runs = Int(1e2)
err_vec = zeros(n_runs + 1)

err_vec[1] = loss_total(ps_all, st_all)
@time for i in 1:n_runs
    local g = gradient(p -> loss_minibatch(p, st_all), ps_all)[1]
    apply!(optim, state, reconstr, ps_all, g)
    err_vec[i + 1] = loss_total(ps_all, st_all)
    println("error is $(err_vec[i+1])")
    if (err_vec[i + 1] - err_vec[i]) / err_vec[i] < -0.1
        optim.η = max(optim.η * 0.9, 1e-6)
    else
        if err_vec[i + 1] < err_vec[i]
            optim.η = min(optim.η * 1.1, 1e-3)
        else
            optim.η = max(optim.η * 0.9, 1e-6)
        end
    end
    println("new learning rate is $(optim.η)")
end

@printf "PSD error: %.5e. " PSD_err
@printf "SAE error: %.5e" err_vec[end]