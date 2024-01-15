"""
The `GrassmannManifold` is based on the `StiefelManifold`
"""
mutable struct GrassmannManifold{T, AT <: AbstractMatrix{T}} <: Manifold{T}
    A::AT
end

function rgrad(Y::GrassmannManifold, e_grad::AbstractMatrix)
    e_grad - Y * (Y' * e_grad)
end

# think about this! This metric may not be the right one!
function metric(Y::GrassmannManifold, Δ₁::AbstractMatrix, Δ₂::AbstractMatrix)
    LinearAlgebra.tr(Δ₁'*(I - Y*inv(Y'*Y)*Y')*Δ₂)
end


function global_section(Y::GrassmannManifold{T}) where T
    N, n = size(Y)
    A = randn(T, N, N-n)
    A - Y * (Y' * A)
    qr!(hcat(Y, A)).Q
end
