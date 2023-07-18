struct ReluActivation <: AbstractActivationFunction end

function (::ReluActivation)(x::T) where {T<:Real}
    max(T(0.),x)
end