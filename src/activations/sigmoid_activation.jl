struct SigmoidActivation <: AbstractActivationFunction end

function (::SigmoidActivation)(x::T) where {T<:Real}
    T(1.)/(T(1.) + exp(-x))
end