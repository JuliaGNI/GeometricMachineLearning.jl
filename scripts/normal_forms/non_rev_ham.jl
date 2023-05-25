using GeometricMachineLearning, Lux, LinearAlgebra
import Random, Zygote, ProgressMeter

τᵋ = Chain(Gradient(2, 16, tanh),
            Gradient(2, 16, tanh; change_q=false),
            Gradient(2, 16, tanh),
            Gradient(2, 16, tanh; change_q=false))

Hᵋ = Chain(Dense(2, 15, tanh), Dense(15, 15, tanh), Dense(15, 1, use_bias=false))

ps_τ, st_τ = Lux.setup(Random.default_rng(), τᵋ)
ps_τ_tuple = Tuple([Tuple(layer) for layer ∈ ps_τ])
ps_F, st_F = Lux.setup(Random.default_rng(), Hᵋ)
ps_F_tuple = Tuple([Tuple(layer) for layer ∈ ps_F])

A = [0. -1.; 1. 0.]
mulA(z::Tuple) = (-z[2], z[1])

Fᵋ(z, ps_F) = -A*(Zygote.gradient(z -> sum(Hᵋ(z, ps_F, st_F)), z)[1])

f(z::AbstractVecOrMat) = [3*z[1]*z[2]^2, -z[2]^3]
f(z::Tuple) = (3*z[1]*z[2]^2, -z[2]^3)
expA(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]

expA(θ, qp::Tuple) = (cos(θ)*qp[1] - sin(θ)*qp[2], cos(θ)*qp[2] + sin(θ)*qp[1])

fτᵋ(z, ps_τ) = τᵋ(z, ps_τ, st_τ) |> f 
#const 
θpoints = range(0, 2*π, 100)[1:end-1]
Πₐfτᵋ(z, ps_τ) = sum([expA(-θ, fτᵋ(expA(θ, z), ps_τ))...] for θ in θpoints) / length(θpoints)

function Jτᵋ(z, ps_τ)
    dq_τq, dp_τq = Zygote.gradient((q,p) -> τᵋ((q,p), ps_τ, st_τ)[1], z[1], z[2])
    dq_τp, dp_τp = Zygote.gradient((q,p) -> τᵋ((q,p), ps_τ, st_τ)[2], z[1], z[2])
    return hcat([dq_τq; dq_τp], [dp_τq; dp_τp])
end
function Jτᵋ(z, ps_τ, dz)
    dq_τq = Zygote.gradient(q -> τᵋ((q, z[2]), ps_τ, st_τ)[1], z[1])[1]
    dp_τq = Zygote.gradient(p -> τᵋ((z[1], p), ps_τ, st_τ)[1], z[2])[1]
    dq_τp = Zygote.gradient(q -> τᵋ((q, z[2]), ps_τ, st_τ)[2], z[1])[1]
    dp_τp = Zygote.gradient(p -> τᵋ((z[1], p), ps_τ, st_τ)[2], z[2])[1]
    return (dq_τq*dz[1] + dp_τq*dz[2], dq_τp*dz[1] + dp_τp*dz[2])
end


#TODO: matrix!!! Jτᵋ outputs a matrix!
function ΠₐJτᵋ(z, ps_τ)
    sum(expA(-θ)*Jτᵋ(expA(θ, z), ps_τ)*expA(θ) for θ in θpoints) / length(θpoints)
end

function init_data_F(ps_τ, batch_size=100)
    qp_points = [(4*rand()-2, 4*rand()-2) for _ in 1:batch_size]
    Π_Jτ = [ΠₐJτᵋ(qp, ps_τ) for qp in qp_points]
    Π_fτ = [Πₐfτᵋ(qp, ps_τ) for qp in qp_points]
    qp_points = [[qp...] for qp in qp_points] # convert to Vectors to feed through Fᵋ
    return qp_points, Π_Jτ, Π_fτ
end

function loss_F(ps_F, qp, Π_Jτ, Π_fτ)
    norm(Π_Jτ*Fᵋ(qp, ps_F) - Π_fτ)
end

function init_data_τ(ps_F, batch_size=100, ε = 0.1)
    qp_points = [[4*rand()-2, 4*rand()-2] for _ in 1:batch_size]
    # convert to Tuples to feed through τᵋ
    Fqp = [Tuple(Fᵋ(qp, ps_F)) for qp in qp_points]
    qp_points = [(qp[1], qp[2]) for qp in qp_points] 
    Aqp = [mulA(qp) for qp in qp_points]
    Aqp_plus_ε_Fqp = [aqp .+ (ε .* fqp) for (aqp, fqp) in zip(Aqp, Fqp)]
    return qp_points, Aqp_plus_ε_Fqp
end

function loss_τ(ps_τ, qp, Aqp_plus_ε_Fqp, ε = 0.1)
    aq, ap = Jτᵋ(qp, ps_τ, Aqp_plus_ε_Fqp)
    bq, bp = mulA(τᵋ(qp, ps_τ, st_τ))
    cq, cp = ε .* fτᵋ(qp, ps_τ)
    sqrt((aq - bq - cq)^2 + (ap - bp - cp)^2)
end

#=
function loss_τ(ps_τ, qp, Aqp, Fqp, ε = 0.1)
    # norm(Jτᵋ(qp, ps_τ, Aqp .+ ε .* Fqp) .- mulA(τᵋ(qp, ps_τ, st_τ)) .- ε .* fτᵋ(qp, ps_τ))
    aq, ap = Jτᵋ(qp, ps_τ, Aqp .+ ε .* Fqp)
    bq, bp = mulA(τᵋ(qp, ps_τ, st_τ))
    cq, cp = ε .* fτᵋ(qp, ps_τ)
    sqrt((aq - bq - cq)^2 + (ap - bp - cp)^2)
end

loss_1(z, ps_τ::Tuple, ps_F::Tuple) = norm(ΠₐJτᵋ(z, ps_τ)*Fᵋ(z, ps_F) - Πₐfτᵋ(z, ps_τ))
loss_2(z, ps_τ::Tuple, ps_F::Tuple, ε=.1) = norm(Jτᵋ(z, ps_τ, mulA(z)) .- mulA(τᵋ(z, ps_τ, st_τ)) .- ε .* (fτᵋ(z, ps_τ) .- Jτᵋ(z, ps_τ, Fᵋ([z...], ps_F))))

loss_3(z, ps_F::Tuple) = norm(Fᵋ(z, ps_F) - rand(2))
loss_4(z, ps_τ::Tuple) = norm(Jτᵋ(z, ps_τ, mulA(z)))
loss_5(z, ps_τ::Tuple) = norm(τᵋ(z, ps_τ, st_τ) .- z)
loss_6(z, ps_τ::Tuple) = norm(mulA(τᵋ(z, ps_τ, st_τ)))
loss_7(z, ps_τ::Tuple) = norm(fτᵋ(z, ps_τ))
=#

o = AdamOptimizer()
cache_F = init_optimizer_cache(Hᵋ, o)
cache_τ = init_optimizer_cache(τᵋ, o)

n_epoch = 1000
batch_size = 100

keys_τ_1 = keys(ps_τ)
keys_τ_2 = [keys(x) for x in values(ps_τ)]

keys_F_1 = keys(ps_F)
keys_F_2 = [keys(x) for x in values(ps_F)]

function update_tuple!(ps_tuple, ps)
    for (layer_tuple, layer) in zip(ps_tuple, ps)
        for (param_tuple, param) in zip(layer_tuple, layer)
            param_tuple .= param
        end
    end
    return ps_tuple
end

function _add!(a::Tuple, b::Tuple)
    for i in 1:length(a)
        _add!(a[i], b[i])
    end
    a
end
function _add!(A::AbstractArray,B::AbstractArray)
    A .+= B
end

function GeometricMachineLearning._add(a::AbstractFloat, b::AbstractFloat)
    a + b
end
function GeometricMachineLearning._add(a::Tuple, b::Tuple)
    c = deepcopy(a)
    _add!(c, b)
end


update_tuple!(ps_F_tuple, ps_F)
update_tuple!(ps_τ_tuple, ps_τ)

loss_F_arr = zeros(n_epoch)
loss_τ_arr = zeros(n_epoch)

#=
ProgressMeter.@showprogress for i in 1:n_epoch
    data_F = init_data_F(ps_τ_tuple, batch_size)
    loss_F_arr[i], g_F = reduce(_add, [Zygote.withgradient(ps -> loss_F(ps, data_F[1][i], data_F[2][i], data_F[3][i]),ps_F_tuple) for i in 1:batch_size])

    g_F = NamedTuple(zip(keys_F_1,[NamedTuple(zip(k, x ./ batch_size)) for (k,x) in zip(keys_F_2, g_F[1])]))
    optimization_step!(o, Hᵋ, ps_F, cache_F, g_F)
    update_tuple!(ps_F_tuple, ps_F)

    data_τ =  init_data_τ(ps_F_tuple, batch_size)
    loss_τ_arr[i], g_τ = reduce(_add, [Zygote.withgradient(ps -> loss_τ(ps, data_τ[1][i], data_τ[2][i]), ps_τ_tuple) for i in 1:batch_size])
    g_τ = NamedTuple(zip(keys_τ_1,[NamedTuple(zip(k, x ./ batch_size)) for (k,x) in zip(keys_τ_2, g_τ[1])]))
    optimization_step!(o, τᵋ, ps_τ, cache_τ, g_τ)
    update_tuple!(ps_τ_tuple, ps_τ)
end
=#

ProgressMeter.@showprogress for i in 1:n_epoch÷2
    data_F = init_data_F(ps_τ_tuple, batch_size)
    loss_F_arr[i], g_F = reduce(_add, [Zygote.withgradient(ps -> loss_F(ps, data_F[1][i], data_F[2][i], data_F[3][i]),ps_F_tuple) for i in 1:batch_size])

    g_F = NamedTuple(zip(keys_F_1,[NamedTuple(zip(k, x ./ batch_size)) for (k,x) in zip(keys_F_2, g_F[1])]))
    optimization_step!(o, Hᵋ, ps_F, cache_F, g_F)
    update_tuple!(ps_F_tuple, ps_F)

    data_τ =  init_data_τ(ps_F_tuple, batch_size)
    loss_τ_arr[i] = sum(loss_τ(ps_τ_tuple, data_τ[1][i], data_τ[2][i]) for i in 1:batch_size)
end


ProgressMeter.@showprogress for i in n_epoch÷2:n_epoch
    data_F = init_data_F(ps_τ_tuple, batch_size)
    loss_F_arr[i] = sum(loss_F(ps_F_tuple, data_F[1][i], data_F[2][i], data_F[3][i]) for i in 1:batch_size)

    data_τ =  init_data_τ(ps_F_tuple, batch_size)
    loss_τ_arr[i], g_τ = reduce(_add, [Zygote.withgradient(ps -> loss_τ(ps, data_τ[1][i], data_τ[2][i]), ps_τ_tuple) for i in 1:batch_size])
    g_τ = NamedTuple(zip(keys_τ_1,[NamedTuple(zip(k, x ./ batch_size)) for (k,x) in zip(keys_τ_2, g_τ[1])]))
    optimization_step!(o, τᵋ, ps_τ, cache_τ, g_τ)
    update_tuple!(ps_τ_tuple, ps_τ)
end

