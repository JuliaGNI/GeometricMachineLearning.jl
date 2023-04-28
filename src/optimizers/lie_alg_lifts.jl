"""
This implements lifts from the base manifold to the Lie algebra.

think about introducing a lift object containing HD and the lifted element of the Lie algebra!!
"""

Ω(Y::StiefelManifold, V::AbstractMatrix) = (I - .5*Y*Y')*V*Y' - Y*V'*(I - .5*Y*Y')

#this is not very efficient - just used for testing purposes
function global_rep_test(Y::StiefelManifold, V::AbstractMatrix)
    B = Ω(Y, V)
    #find complement for global section
    N, n = size(Y)
    A = randn(N, N-n)
    A = A - Y*Y'*A
    HD = HouseDecom(A)
    QTB = HD'(B)
    QTBY = QTB*Y 
    B = hcat(vcat(Y'*B*Y, QTBY), vcat(-QTBY', -HD'(QTB')))
    return (HD, B)
end

#does the same as the above function - with the difference in the output format!!!! (no overhead!)
function global_rep(Y::StiefelManifold, V::AbstractMatrix)
    B = Ω(Y, V)
    #find complement for global section
    N, n = size(Y)
    A = randn(N, N-n)
    A = A - Y*Y'*A
    HD = HouseDecom(A)
    QTB = HD'(B)
    B = StiefelLieAlgHorMatrix(SkewSymMatrix(Y'*B*Y), QTB*Y, N, n)
    return (HD, B)
end

#A is a QR decomposition - make this a specific type!
function apply_projection(Y::StiefelManifold, HD::HouseDecom, B::StiefelLieAlgHorMatrix)
    Y*B.A + HD(B.B)
end