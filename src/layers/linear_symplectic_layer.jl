
# residual layer that changes p
const LinearSymplecticLayerP = LinearFeedForwardLayer

function LinearSymplecticLayerP(W, gradient)
    S = SymmetricBlockIdentityLowerMatrix(W)
    b = ZeroVector(eltype(W), 2*length(axes(W,1)))
    LinearFeedForwardLayer(S, b, gradient)
end

# residual layer that changes q
const LinearSymplecticLayerQ = LinearFeedForwardLayer(W, b, gradient)

function LinearSymplecticLayerQ(W, gradient)
    S = SymmetricBlockIdentityUpperMatrix(W)
    b = ZeroVector(eltype(W), 2*length(axes(W,1)))
    LinearFeedForwardLayer(S, b, gradient)
end
