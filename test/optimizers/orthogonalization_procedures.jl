"""
This implements some tests for orthogonalization_procedures: Gram-Schmidt & Householder reflections.
    (For the regular and the symplectic Stiefel case.)
"""


function gram_schmidt_test(N)
    A = randn(N, N)
    A_orth = gram_schmidt(A)
    check(StiefelManifold(A_orth))
end

#the matrices are sampled using the Cayley transform at the moment! 
#implement sampling in the type SymplecticLieAlgMatrix!
function sympl_gram_schmidt_test(N)
    J = SymplecticMatrix(N)
    A = randn(2*N, 2*N)
    A = SymplecticLieAlgMatrix(A)
    A = Cayley(A)
    check(A)
    #add perturbation
    A += 0.05*randn(2*N, 2*N)
    A_sympl = sympl_gram_schmidt(A, J)
    check(SymplecticStiefelManifold(A_sympl))
end