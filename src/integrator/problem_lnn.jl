# This file contains the functions to create the corresponding problem to lnn which is LODEProblem

function LNNProblem(nn::NeuralNetwork{<:LagrangianNeuralNetwork}, tspan::Tuple, tstep::Real, ics...; kwargs...)
    
    Lₛₚₗᵢₜ(q,v) = sum(nn([q...,v...]))

    ∇Lₛₚₗᵢₜ(q,v) = Zygote.gradient(Lₛₚₗᵢₜ, q, v)
    
    function p(p, t, q, v, params)
        p .= ∇Lₛₚₗᵢₜ(q,v)[2]
    end
    
    function f(f, t, q, v, params)
        f .=  ∇Lₛₚₗᵢₜ(q,v)[1]
    end
    
    function ω(ω, t, q, v, params)
        n_dim = length(q)
        I = Diagonal(ones(n_dim))
        Z = zeros(n_dim,n_dim)
        ω .= [Z I;-I Z]
    end
    
    function lagrangian(t, q, v, params)
        Lₛₚₗᵢₜ(q,v)
    end

    LODEProblem(p, f, ω, lagrangian, tspan, tstep, ics...; kwargs...)
end
