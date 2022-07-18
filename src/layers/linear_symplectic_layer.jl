
# residual layer that changes p
const LinearSymplecticLayerP{DT, N, M, WT <: BlockIdentityLowerMatrix{DT}, GT} = LinearFeedForwardLayer{DT, N, M, WT, <: ZeroVector, GT}

function LinearSymplecticLayerP(W, gradient)
    S = SymmetricBlockIdentityLowerMatrix(W)
    b = ZeroVector(eltype(W), 2*length(axes(W,1)))
    LinearFeedForwardLayer(S, b, gradient)
end

# residual layer that changes q
const LinearSymplecticLayerQ{DT, N, M, WT <: BlockIdentityUpperMatrix{DT}, GT} = LinearFeedForwardLayer{DT, N, M, WT, <: ZeroVector, GT}

function LinearSymplecticLayerQ(W, gradient)
    S = SymmetricBlockIdentityUpperMatrix(W)
    b = ZeroVector(eltype(W), 2*length(axes(W,1)))
    LinearFeedForwardLayer(S, b, gradient)
end
