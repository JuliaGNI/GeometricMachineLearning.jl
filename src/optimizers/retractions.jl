"""
This implements some basic retractions.
"""

function Cayley(A::AbstractMatrix)
    N = size(A)[1] 
    (I(N) - .5*A)*inv(I(N) - .5*A)
end

function Cayley(A::SkewSymMatrix)
    StiefelManifold(Cayley(Matrix(B)))
end

function Cayley(A::SymplecticLieAlgMatrix)
    SymplecticStiefelManifold(Cayley(Matrix(A)))
end

function Exp(A::SkewSymMatrix)
    StiefelManifold(exp(A))
end

function Exp(A::SymplecticLieAlgMatrix)
    SymplecticStiefelManifold(exp(A))
end 

#geodesic retrations
function Geo(A::SkewSymMatrix)
end

function Geo(A::SymplecticLieAlgMatrix)
end

#function Cayley(U::StiefelManifold,::SymplecticLieAlgHorMatrix)
#end 