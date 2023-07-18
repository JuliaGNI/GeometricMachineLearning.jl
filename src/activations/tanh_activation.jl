struct TanhActivation <: AbstractActivationFunction end

(::TanhActivation)(x::Real) = tanh(x)