@doc raw"""
This is a PSD-like layer used for symplectic autoencoders. 
One layer has the following shape:

```math
A = \begin{bmatrix} \Phi & \mathbb{O} \\ \mathbb{O} & \Phi \end{bmatrix},
```
where $\Phi$ is an element of the Stiefel manifold $St(n, N)$.

The constructor of PSDLayer is called by `PSDLayer(M, N; retraction=retraction)`: 
- `M` is the input dimension.
- `N` is the output dimension. 
- `retraction` is an instance of a struct with supertype `AbstractRetraction`. The only options at the moment are `Geodesic()` and `Cayley()`.
"""
struct PSDLayer{M, N, Retraction} <: LayerWithManifold{M, N, Retraction} end

default_retr = Geodesic()
function PSDLayer(M::Integer, N::Integer; retraction=default_retr)
    @assert iseven(M)
    @assert iseven(N)
    PSDLayer{M, N, typeof(retraction)}()
end

function parameterlength(::PSDLayer{M, N}) where {M, N}
    M÷2*(N÷2 - (M÷2+1)÷2)
end 

function initialparameters(::PSDLayer{M, N}, backend::KernelAbstractions.Backend, T::Type; rng::AbstractRNG=Random.default_rng()) where {M, N}
    (weight =  N > M ? rand(backend, rng, StiefelManifold{T}, N÷2, M÷2) : rand(backend, rng, StiefelManifold{T}, M÷2, N÷2), )
end

function (::PSDLayer{M, N})(x::AbstractArray, ps::NamedTuple) where {M, N}
    dim = size(x, 1)
    @assert dim == M 

    qp = assign_q_and_p(x, dim÷2)
    N > M ? vcat(custom_mat_mul(ps.weight, qp.q), custom_mat_mul(ps.weight, qp.p)) : vcat(custom_mat_mul(ps.weight', qp.q), custom_mat_mul(ps.weight', qp.p))
end