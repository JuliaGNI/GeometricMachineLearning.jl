
struct IdentityActivation <: ActivationFunction end

(::IdentityActivation)(x) = x
