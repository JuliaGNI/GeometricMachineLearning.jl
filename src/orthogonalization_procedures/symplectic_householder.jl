N = 5
M = 4

A = rand(2*N, 2*M)
J = SymplecticMatrix(N)

function zero_rot(A)
    N, M = size(A).÷2
    J = SymplecticMatrix(N)
    α = A[:,1]*J*A[:,1:M]
    v₁ = A[:,1] - A[:,1]
end
