using GeometricMachineLearning, Lux, LinearAlgebra
import Random, Zygote

τᵋ = Chain(Gradient(2, 10, tanh),
            Gradient(2, 10, tanh; change_q=false))

Hᵋ = Chain(Dense(2, 10, tanh), Dense(10, 1, use_bias=false))

ps_τ, st_τ = Lux.setup(Random.default_rng(), τᵋ)
ps_F, st_F = Lux.setup(Random.default_rng(), Hᵋ) 

Fᵋ(z, ps_F) = SymplecticMatrix(1)*(Zygote.gradient(z -> sum(Hᵋ(z, ps_F, st_F)[1]), z)[1])

f(z) = [3*z[1]*z[2]^2, -z[2]^3]
A = [0. -1.; 1. 0.]
expA(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]

fτᵋ(z, ps_τ) = τᵋ(z, ps_τ, st_τ)[1] |> f 
#const 
θpoints = range(0, 2*π, 100)[1:end-1]
function Πₐfτᵋ(z, ps_τ)
    output = zero(z)
    for θ in θpoints
       output += expA(-θ)*fτᵋ(expA(θ)*z, ps_τ)        
    end
    output/(2*π*length(θpoints))
end

Jτᵋ(z, ps_τ) = Zygote.jacobian(z -> τᵋ(z, ps_τ, st_τ)[1], z)[1]

function ΠₐJτᵋ(z, ps_τ) 
    output = zeros(2,2)
    for θ in θpoints
        output += expA(-θ)*Jτᵋ(expA(θ)*z, ps_τ)
        #display(output)
    end
    output/(2*π*length(θpoints))
end

loss_1(z, ps_τ, ps_F) = norm(ΠₐJτᵋ(z, ps_τ)*Fᵋ(z, ps_F) - Πₐfτᵋ(z, ps_τ))
loss_2(z, ps_τ, ps_F, ε=.1) = norm(Jτᵋ(z, ps_τ)*A*z - A*(τᵋ(z, ps_τ, st_τ)[1]) - ε*(fτᵋ(z, ps_τ) - Jτᵋ(z, ps_τ)*Fᵋ(z, ps_F)))