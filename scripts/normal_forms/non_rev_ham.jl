using GeometricMachineLearning, Lux, LinearAlgebra
import Random, Zygote

τᵋ = Chain(Gradient(2, 10, tanh),
            Gradient(2, 10, tanh; change_q=false))

Hᵋ = Chain(Dense(2, 10, tanh), Dense(10, 1, use_bias=false))

ps_τ, st_τ = Lux.setup(Random.default_rng(), τᵋ)
ps_F, st_F = Lux.setup(Random.default_rng(), Hᵋ) 

A = [0. -1.; 1. 0.]

Fᵋ(z, ps_F) = -A*(Zygote.gradient(z -> sum(Hᵋ(z, ps_F, st_F)), z)[1])

f(z) = [3*z[1]*z[2]^2, -z[2]^3]
expA(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]

expA(θ, qp::Tuple) = Tuple(cos(θ)*qp[1] - sin(θ)*qp[2], cos(θ)*qp[2] + sin(θ)*qp[1])
expAr(θ, qp::Tuple) = 

fτᵋ(z, ps_τ) = τᵋ(z, ps_τ, st_τ) |> f 
#const 
θpoints = range(0, 2*π, 100)[1:end-1]
Πₐfτᵋ(z, ps_τ) = sum(expA(-θ, fτᵋ(expA(θ, z), ps_τ)) for θ in θpoints) / length(θpoints)

Jτᵋ(z, ps_τ) = Zygote.jacobian(z -> τᵋ(z, ps_τ, st_τ), z)[1]

#TODO: matrix!!! Jτᵋ outputs a matrix!
function ΠₐJτᵋ(z, ps_τ) 
    sum(expA(-θ)*Jτᵋ(expA(θ, z), ps_τ)*expA(θ) for θ in θpoints) / length(θpoints)
end

function training_step(ps_τ, ps_F, batch_size=100)
    qp_points = rand((batch_size,2))
    qp_points = [(qp[i,1], qp[i,2]) for i in 1:batch_size]
    Π_Jτ = = 
    Π_fτ
end


loss_1(z, ps_τ::Tuple, ps_F::Tuple) = norm(ΠₐJτᵋ(z, ps_τ)*Fᵋ(z, ps_F) - Πₐfτᵋ(z, ps_τ))
loss_2(z, ps_τ::Tuple, ps_F::Tuple, ε=.1) = norm(Jτᵋ(z, ps_τ)*A*z - A*(τᵋ(z, ps_τ, st_τ)) - ε*(fτᵋ(z, ps_τ) - Jτᵋ(z, ps_τ)*Fᵋ(z, ps_F)))

loss_3(z, ps_F::Tuple) = norm(Fᵋ(z, ps_F) - rand(2))