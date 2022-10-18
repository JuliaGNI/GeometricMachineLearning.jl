
struct IdentityActivation <: AbstractActivationFunction end

(::IdentityActivation)(x) = x
