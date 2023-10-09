"""
TODO: 
- Rename this file to "analytic solution". 
"""

using Plots, ForwardDiff

function h(x::T) where T
    if T(0) ≤ x ≤ T(1)
        T(1) - T(1.5)*x^2 + T(.75)*x^3 
    elseif T(1) < x ≤ 2
        T(.25)*(T(2) - x)^3
    else
        T(0)
    end 
end

function s(ξ, μ::T) where T
    T(4) / μ * abs(ξ + T(.5)*(T(1) - μ))
end

u₀(ξ, μ) = h(s(ξ, μ))
u(t, ξ, μ) = u₀(ξ .- μ * t, μ)
p(t, ξ, μ) = ForwardDiff.derivative(t -> u(t,ξ,μ), t)
function p(t::T, ξ::AbstractVector{T}, μ::T) where T
    p_closure(ξ) = p(t, ξ, μ)
    p_closure.(ξ)
end

function s(ξ::AbstractVector{T}, μ::T) where T 
    s_closure(ξ_scal) = s(ξ_scal, μ)
    s_closure.(ξ)
end

function u₀(ξ::AbstractVector{T}, μ::T) where T 
    h.(s(ξ, μ))
end

function get_domain(T=Float32, spacing=T(.01), time_step=T(0.01))
    Ω = T(-.5):spacing:T(.5)
    I = T(0):time_step:T(1)
    Ω, I 
end

function plot_time_evolution(T=Float32; spacing=T(.01), time_step=T(0.25), μ=T(.3))
    Ω, I = get_domain(T, spacing, time_step)
    curves = zeros(T, length(Ω), length(I))
    curves_p = zeros(T, length(Ω), length(I))

    for it in zip(axes(I, 1), I) 
        i = it[1]; t = it[2]
        curves[1:length(Ω), i] = u(t, Ω, μ)
        curves_p[1:length(Ω), i] = p(t, Ω, μ)
    end
    curves, curves_p, plot(Ω, curves, layout=(length(I), 1)), plot(Ω, curves_p, layout=(length(I), 1))
end

function generate_data(T=Float32; spacing=T(.01), time_step=T(0.01), μ_collection=T(5/12):T(.1):T(5/6))
    Ω, I = get_domain(T, spacing, time_step)
    curves = zeros(T, 2*length(Ω), length(μ_collection), length(I))

    for it in zip(axes(I, 1), I)
        i = it[1]; t = it[2]
        for it2 in zip(axes(μ_collection, 1), μ_collection)
            j = it2[1]; μ = it2[2]
            curves[1:length(Ω), j, i] = u(t, Ω, μ)
            curves[(length(Ω)+1):2*length(Ω), j, i] = p(t, Ω, μ)
        end
    end
    curves
end

function analytic_solution(T=Float64; N::Int=2048, n_time_steps=4000, n_μ::Int=8)
    μ_spacing = (T(5/6) - T(5/12))/(n_μ - 1)
    μ_collection = T(5/12):μ_spacing:T(5/6)
    spacing = T(1/(N-1))
    time_step = T(1/(n_time_steps-1))
    generate_data(T; spacing=spacing, time_step=time_step, μ_collection=μ_collection)
end