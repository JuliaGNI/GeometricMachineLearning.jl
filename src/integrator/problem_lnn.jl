# This file contains the functions to create the corresponding problem to lnn which is LODEProblem

function LNNProblem(nn::NeuralNetwork{<:LagrangianNeuralNetwork}, g, tspan::Tuple, tstep::Real, ics::NamedTuple; kwargs...)
    
    Lₛₚₗᵢₜ(q,v) = nn([q...,v...])

    ∇Lₛₚₗᵢₜ(q,v) = Zygote.gradient(Lₛₚₗᵢₜ, q, v)
    
    function p(p, t, q, v, params)
        p .= ∇Lₛₚₗᵢₜ(q,v)[2]
    end
    
    function f(f, t, q, v, params)
        p .=  ∇Lₛₚₗᵢₜ(q,v)[1]
    end
    
    function ω(f, t, q, v, params)
        n_dim = length(q)
        I = Diagonal(ones(n_dim))
        Z = zeros(n_dim,n_dim)
        ω = [Z I;-I Z]
    end
    
    function lagrangian(t, q, v, params)
        Lₛₚₗᵢₜ(q,v)
    end
    LODEProblem(p, f, g, ω, lagrangian, tspan, tstep, ics; kwargs...)
end

function LNNProblem(nn::NeuralNetwork{<:LagrangianNeuralNetwork}, tspan::Tuple, tstep::Real, ics::NamedTuple; kwargs...)
    LNNProblem(nn, GeometricEquations._lode_default_g, tspan, tstep, ics; kwargs...)
end

function LNNProblem(nn::NeuralNetwork{<:LagrangianNeuralNetwork}, g, tspan::Tuple, tstep::Real, q₀::State, p₀::State, λ₀::State = zero(q₀); kwargs...)
    ics = (q = q₀, p = p₀, λ = λ₀)
    LNNProblem(nn, g, tspan, tstep, ics; kwargs...)
end

function LNNProblem(nn::NeuralNetwork{<:LagrangianNeuralNetwork}, tspan::Tuple, tstep::Real, q₀::State, p₀::State, λ₀::State = zero(q₀); kwargs...)
    LNNProblem(nn, GeometricEquations._lode_default_g, tspan, tstep, q₀, p₀, λ₀; kwargs...)
end