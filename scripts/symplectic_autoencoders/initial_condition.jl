using ForwardDiff, OffsetArrays

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

function p₀(ξ::T, μ::T) where T 
    - μ * ForwardDiff.derivative(ξ -> u₀(ξ, μ), ξ)
end

function p₀(ξ::AbstractVector{T}, μ::T) where T 
    p₀_closure(ξ) = p₀(ξ, μ)
    p₀_closure.(ξ)
end

function get_initial_condition(μ::T, Δx::T) where T
    Ω = T(-.5):Δx:T(.5)
    (q=OffsetArray(u₀(Ω,μ), OffsetArrays.Origin(0)), p=OffsetArray(p₀(Ω,μ), OffsetArrays.Origin(0)))
end

function get_initial_condition2(μ::T, Δx::T) where T 
    Ω = T(-.5):Δx:T(0.5)
    (q=OffsetArray(u₀(Ω,μ), OffsetArrays.Origin(0)), p=OffsetArray(zero(Ω), OffsetArrays.Origin(0)))
end 

"""
If you call this function with a Float and an Integer, then the integer will be interpreted as the number of nodes, i.e. degrees of freedom.
"""
function get_initial_condition(μ::T, Ñ::Integer) where T 
    get_initial_condition(μ, T(1/(Ñ+1)))
end

function get_initial_condition2(μ::T, Ñ::Integer) where T 
    get_initial_condition2(μ, T(1/(Ñ+1)))
end

function get_initial_condition_vector(μ::T, Ñ::Integer) where T 
    ics_offset = get_initial_condition(μ, Ñ)
    vcat(ics_offset.q.parent, ics_offset.p.parent)
end

function get_initial_condition_vector2(μ::T, Ñ::Integer) where T
    ics_offset = get_initial_condition2(μ, Ñ)
    vcat(ics_offset.q.parent, ics_offset.p.parent)
end