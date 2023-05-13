#start index indicates if the orthonormalization is started at positon 0
function gram_schmidt!(A::AbstractMatrix, start=1)
    N = size(A)[1]
    n = size(A)[2]
    @assert n ≤ N 
    
    for i in start:n
        vec = A[1:N,i]
        for j in 1:(i-1)
            vec = vec - vec'*A[1:N,j]*A[1:N,j]
        end
        #print("GS: ",norm(vec),"\n")
        A[1:N, i] = norm(vec)^-1*vec 
    end
end 

#this "normalizes" 2 vectors according to the symplectic form (e,f) -> e'*J*f
function normalize(e::AbstractVector ,f::AbstractVector , J::AbstractMatrix)
    fac = e'*J*f
    #print("SGS: ",fac,"\n")
    (sign(fac)/sqrt(abs(fac))*e, 1/sqrt(abs(fac))*f)
end

function sympl_gram_schmidt!(A::AbstractMatrix, J::AbstractMatrix, start=1)
    N = size(A)[1]
    n = size(A)[2]
    @assert n ≤ N 
    @assert iseven(N)
    @assert iseven(n)
    N ÷= 2 
    n ÷= 2

    for i in start:n
        vec₁ = A[1:(2*N),i]
        vec₂ = A[1:(2*N),n+i]
        for j in 1:(i-1)
            vec₁ = vec₁ - (A[1:(2*N),j]'*J*vec₁)*A[1:(2*N),n+j] - (vec₁'*J*A[1:(2*N),n+j])*A[1:(2*N),j]
            vec₂ = vec₂ - (A[1:(2*N),j]'*J*vec₂)*A[1:(2*N),n+j] - (vec₂'*J*A[1:(2*N),n+j])*A[1:(2*N),j]
        end
        A[1:(2*N),i], A[1:(2*N),n+i]  =  normalize(vec₁, vec₂, J)
    end
end 

function gram_schmidt(A::AbstractMatrix, start=1)
    B = deepcopy(A)
    gram_schmidt!(B, start)
    B
end

function sympl_gram_schmidt(A::AbstractMatrix, J::AbstractMatrix, start=1)
    B = deepcopy(A)
    sympl_gram_schmidt!(B, J, start)
    B
end
