using Plots 

function h(x::T) where T
    if T(0) ≤ x ≤ T(1)
        T(1) - T(1.5)*x^2 + T(.75)*x^3 
    elseif T(1) < x ≤ 2
        T(.25)*(T(2) - x)^3
    else
        T(0)
    end 
end

function s(ξ::T, μ::T) where T 
    T(4) / μ * abs(ξ + T(.5)*(T(1) - μ))
end

u₀(ξ, μ) = h∘s(ξ, μ)
u(t, ξ, μ) = u₀(ξ .- μ * t, μ)

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

    for it in zip(axes(I, 1), I) 
        i = it[1]; t = it[2]
        curves[:, i] = u(t, Ω, μ)
    end
    curves, plot(Ω, curves, layout=(length(I), 1))
end

#= 
μ = Float32(5/12)
data, p = plot_time_evolution(;μ=μ)
png(p, "data_for_μ="*string(μ))
=#

function generate_data(T=Float32; spacing=T(.01), time_step=T(0.01), μ_collection=T(5/12):T(.1):T(5/6))
    Ω, I = get_domain(T, spacing, time_step)
    curves = zeros(T, length(Ω), length(μ_collection), length(I))

    for it in zip(axes(I, 1), I)
        i = it[1]; t = it[2]
        for it2 in zip(axes(μ_collection, 1), μ_collection)
            j = it2[1]; μ = it2[2]
            curves[:, j, i] = u(t, Ω, μ)
        end
    end
    curves
end