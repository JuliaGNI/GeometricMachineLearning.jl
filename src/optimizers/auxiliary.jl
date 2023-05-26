"""
This implements exponential and inverse mappings.
"""

#computes A^-1(exp(A) - I)
function 𝔄(A::AbstractMatrix{T}) where T
    B = one(A)
    C = one(A)
    B_temp = zero(A)
    i = 2
    while norm(B) > eps(T)
        mul!(B_temp, B, A)
        B .= B_temp
        rmul!(B, T(inv(i)))
        C += B
        i += 1 
    end
    #print("\nNumber of iterations is: ", i, "\n")
    C
end

function 𝔄exp(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where T
    I + X*A_PS(Y'*X)*Y'
end