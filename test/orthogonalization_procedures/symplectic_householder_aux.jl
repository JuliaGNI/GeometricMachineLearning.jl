using GeometricMachineLearning
using GeometricMachineLearning: J 
using Test 
using LinearAlgebra

function symplectic_hous_mat(v::AbstractVector{T}, c::T) where {T}
    N2 = length(v)
    @assert iseven(N2)
    vJ = vcat(-v[N2÷2+1:N2], v[1:N2÷2])
    one(ones(T, N2, N2)) + c*v*vJ'
end

function orthogonal_hous_mat(v::AbstractVector{T}) where {T}
    N2 = length(v)
    one(ones(T, N2, N2)) - 2*v*v'/(v'*v)
end

function test_symplecticity_preservation()
    S = one(ones(T, 2*N, 2*N))
    for i in 1:num_factors 
        v = rand(T, 2*N)
        c = scal*randn(T)
        S = symplectic_hous_mat(v, c)*S
    end
    norm(S'*J_mat*S - J_mat)
end

function test_orthogonality_preservation()
    Q = one(ones(T, 2*N, 2*N))
    Id = copy(Q)
    for i in 1:num_factors 
        v = rand(T, 2*N)
        Q = orthogonal_hous_mat(v)*Q
    end
    norm(Q'*Q - Id)
end


num_factors = 10
N = 100
T = Float32
scal = T(.1)
J_mat = SymplecticPotential(T, N)

print(
    test_symplecticity_preservation(),
    " ",
    test_orthogonality_preservation()
)