
# residual layer that changes p
const LinearSympNetLayerP = LinearFeedForwardLayer

function LinearSympNetLayerP(W, gradient)
    S = SymmetricBlockIdentityLowerMatrix(W)
    b = ZeroVector(eltype(W), 2*length(axes(W,1)))
    LinearFeedForwardLayer(S, b, gradient)
end

# residual layer that changes q
const LinearSympNetLayerQ = LinearFeedForwardLayer(W, b, gradient)

function LinearSympNetLayerQ(W, gradient)
    S = SymmetricBlockIdentityUpperMatrix(W)
    b = ZeroVector(eltype(W), 2*length(axes(W,1)))
    LinearFeedForwardLayer(S, b, gradient)
end
