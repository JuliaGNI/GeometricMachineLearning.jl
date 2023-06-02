using GeometricMachineLearning
using HDF5
using LinearAlgebra
using Lux
using Printf
using Random
using Zygote
using ProgressMeter

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

relu(x) = max.(0, x)
Ψ_enc = Chain(Gradient(2 * n, 4 * n, relu; change_q = true),
              Gradient(2 * n, 4 * n, relu; change_q = false),
              PSDLayer(2 * n, 100; inverse = true),
              Gradient(100, 200, relu; change_q = true),
              Gradient(100, 200, relu; change_q = false),
              PSDLayer(100, 50; inverse = true),
              Gradient(50, 200, relu; change_q = true),
              Gradient(50, 200, relu; change_q = false),
              PSDLayer(50, 2 * m; inverse = true),
              Gradient(2 * m, 4 * m, relu; change_q = false),
              Gradient(2 * m, 4 * m, relu; change_q = true))
Ψ_dec = Chain(Gradient(2 * m, 4 * m, relu; change_q = false),
              Gradient(2 * m, 4 * m, relu; change_q = true),
              PSDLayer(50, 2 * m),
              Gradient(50, 200, relu; change_q = false),
              Gradient(50, 200, relu; change_q = true),
              PSDLayer(100, 50),
              Gradient(100, 200, relu; change_q = false),
              Gradient(100, 200, relu; change_q = true),
              PSDLayer(2 * n, 100),
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

#optim = StandardOptimizer(1e-3)
optim = AdamOptimizer()
#TODO: dispatch over the optimizer
cache = init_optimizer_cache(reconstr, optim)
n_runs = Int(1e3)
err_vec = zeros(n_runs + 1)

err_vec[1] = loss_total(ps_all, st_all)
@showprogress for i in 1:n_runs
    g = Zygote.gradient(p -> loss_minibatch(p, st_all), ps_all)[1]
    #apply!(optim, nothing, reconstr, ps_all, g)
    optimization_step!(optim, reconstr, ps_all, cache, g)
    err_vec[i + 1] = loss_total(ps_all, st_all)
    println("error is $(err_vec[i+1])")
end

@printf "PSD error: %.5e. " PSD_err
@printf "SAE error: %.5e\n" err_vec[end]

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