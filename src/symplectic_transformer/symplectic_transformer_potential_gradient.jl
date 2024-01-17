@doc raw"""
The gradient of the symplectic transformer potential. This is used to build the symplectic attention.

We here use that the Jacobian of the softmax function is (where we write ``\mathrm{softmax}(x) =: s(x)``): 
```math
\frac{\partial{}s_j}{\partial{}x_i} = s_i\delta_{ij} - s_is_j.
```

Further we write ``x_{\bullet{}k} = [\mathrm{softmax}(Z^TAZ)]_{\bullet{}k}``, i.e. ``x_{\bullet{}k}`` is the ``k``-th column of ``\Lambda(Z):=\mathrm{softmax}(Z^TAZ)``. We also write ``[x_{\bullet{}k}]_i = x_{ik}`` and hence: 

```math
\frac{\partial{}x_{ik}}{\partial{}z_{m\ell}} = \sum_{h}(\delta_{i\ell}a_{mh}z_{hk} + \delta_{k\ell}a_{hm}z_{hi}).
```

For the total gradient we have: 
```math
\frac{\partial{}F}{\partial{}z_{m\ell}} = \frac{1}{2}\sum_ks_{\ell{}k}z_{mk} + \frac{1}{2}\sum_jz_{mj}s_{j\ell} + \sum_{ijk}z_{ij}\frac{\partial{}s_{jk}}{\partial{}z_{m\ell}}z_{ik}.
```
"""
function symplectic_transformer_potential_gradient(Z::AbstractArray{T}, A::AbstractMatrix{T}) where T 
    S = _compute_softmax(Z, A)
    T(.5) * _mul(
                Z, 
                (S + _transpose(S))
                ) + SDerArrayC(Z, A) 
end 

@doc raw"""
Computes the softmax term in the attention mechanism, i.e. 
```math
S = \mathrm{softmax}(Z^TAZ)
```
"""
function _compute_softmax(Z::AbstractArray{T}, A::AbstractMatrix{T}) where T
    softmax(
        _mul(
            _transpose_mat_mul(Z, A), 
            Z)
    )
end

_transpose_mat_mul(Z::AbstractArray{T, 3}, A::AbstractMatrix{T}) where T = tensor_transpose_mat_mul(Z, A)
_transpose_mat_mul(Z::AbstractMatrix, A::AbstractMatrix) = Z' * A
_mul(Z::AbstractArray{T, 3}, Ẑ::AbstractArray{T, 3}) where T = tensor_tensor_mul(Z, Ẑ)
_mul(Z::AbstractMatrix, Ẑ::AbstractMatrix) = Z * Ẑ
_mul(A::AbstractMatrix{T}, Z::AbstractArray{T, 3}) where T = mat_tensor_mul(A, Z)
_transpose(S::AbstractArray{T, 3}) where T = tensor_transpose(S)
_transpose(S::AbstractMatrix) = S' 

@doc raw"""
This computes the term

```math
\sum_{ijk}z_{ij}\frac{\partial{}s_{jk}}{\partial{}z_{m\ell}}z_{ik}
```
"""
function s_derivative_term(Z::AbstractArray{T, 3}, A::AbstractMatrix{T}, S::AbstractArray{T, 3}, m::Int, l::Int, f::Int) where {T} 
    n, time_steps = size(Z)
    AZ = _mul(A, Z) 
    ATZ = _mul(A', Z)

    T(.5) * sum(Z[i, k, f] * (
                            compute_first_term(Z, S, AZ, i, k, m, l, f) - 
                            compute_second_term(Z, S, AZ, i, k, m, l, f) + 
                            compute_third_term(Z, S, ATZ, i, k, m, l, f) - 
                            compute_fourth_term(Z, S, ATZ, i, k, m, l, f) ) for i=1:n, k=1:time_steps)
end 

function s_derivative_term(Z::AbstractMatrix{T}, A::AbstractMatrix{T}, S::AbstractMatrix{T}, m::Int, l::Int) where {T} 
    n, time_steps = size(Z)
    AZ = _mul(A, Z) 
    ATZ = _mul(A', Z)

    T(.5) * sum(Z[i, k] * (
                            compute_first_term(Z, S, AZ, i, k, m, l) - 
                            compute_second_term(Z, S, AZ, i, k, m, l) + 
                            compute_third_term(Z, S, ATZ, i, k, m, l) - 
                            compute_fourth_term(Z, S, ATZ, i, k, m, l) ) for i=1:n, k=1:time_steps)
end 

@doc raw"""
Realizes the matrix (or a tensor if the input array ``Z`` is a tensor):

```math
[Matrix]_{m\ell} = \frac{1}{2}\sum_{ijk}z_{ij}\frac{\partial{}s_{jk}}{\partial{}z_{m\ell}}z_{ik}
```
"""
struct SDerMatrix{T, AT, BT} <: AbstractMatrix{T} 
    Z::AT 
    A::BT 
    S::AT 
end

struct SDerTensor{T, AT, BT} <: AbstractArray{T, 3} 
    Z::AT 
    A::BT 
    S::AT 
end

const SDerArray = Union{SDerMatrix, SDerTensor}

function SDerMatrix(Z::AT, A::BT, S::AT) where {T, AT<:AbstractMatrix{T}, BT<:AbstractMatrix{T}} 
    SDerMatrix{T, AT, BT}(Z, A, S)
end 

SDerMatrix(Z, A) = SDerMatrix(Z, A, _compute_softmax(Z, A))

function SDerTensor(Z::AT, A::BT, S::AT) where {T, AT<:AbstractArray{T, 3}, BT<:AbstractMatrix{T}}
    SDerTensor{T, AT, BT}(Z, A, S)
end

SDerTensor(Z, A) = SDerTensor(Z, A, _compute_softmax(Z, A))

SDerArrayC(Z::AT, A::BT, S::AT) where {T, AT<:AbstractMatrix{T}, BT<:AbstractMatrix{T}} = SDerMatrix(Z, A, S)
SDerArrayC(Z::AT, A::BT, S::AT) where {T, AT<:AbstractArray{T, 3}, BT<:AbstractMatrix{T}} = SDerTensor(Z, A, S)
SDerArrayC(Z, A) = SDerArrayC(Z, A, _compute_softmax(Z, A))

Base.getindex(ZSZ::SDerMatrix, m::Int, l::Int) = s_derivative_term(ZSZ.Z, ZSZ.A, ZSZ.S, m, l)
Base.getindex(ZSZ::SDerTensor, m::Int, l::Int, f::Int) = s_derivative_term(ZSZ.Z, ZSZ.A, ZSZ.S, m, l, f)

Base.size(ZSZ::SDerArray) = size(ZSZ.Z)

@doc raw"""
This computes the term 

```math
z_{i\ell}s_{ell}(x_{\bullet{}k})[AZ]_{mk}
```
"""
function compute_first_term(Z::AbstractMatrix, S::AbstractMatrix, AZ::AbstractMatrix, i::Int, k::Int, m::Int, l::Int)
    Z[i, l] * S[l, k] * AZ[m, k]
end

function compute_first_term(Z::AbstractArray{T, 3}, S::AbstractArray{T, 3}, AZ::AbstractArray{T, 3}, i::Int, k::Int, m::Int, l::Int, f::Int) where T 
    Z[i, l, f] * S[l, k, f] * AZ[m, k, f]
end

@doc raw"""
This computes the term 

```math
\sum_j z_{ij}s_{j}(x_{\bullet{}k})s_{\ell}(x_{\bullet{}k})[AZ]_{mk}
```
"""
function compute_second_term(Z::AbstractMatrix, S::AbstractMatrix, AZ::AbstractMatrix, i::Int, k::Int, m::Int, l::Int)
    time_steps = size(Z, 2)
    sum(Z[i, j] * S[j, k] * S[l, k] * AZ[m, k] for j=1:time_steps)
end

function compute_second_term(Z::AbstractArray{T, 3}, S::AbstractArray{T, 3}, AZ::AbstractArray{T, 3}, i::Int, k::Int, m::Int, l::Int, f::Int) where T
    time_steps = size(Z, 2)
    sum(Z[i, j, f] * S[j, k, f] * S[l, k, f] * AZ[m, k, f] for j=1:time_steps)
end

@doc raw"""
This computes the term 

```math
z_{i\ell}s_k(x_{\bullet{}\ell})[A^TZ]_{mk}
```
"""
function compute_third_term(Z::AbstractMatrix, S::AbstractMatrix, ATZ::AbstractMatrix, i::Int, k::Int, m::Int, l::Int)
    Z[i, l] * S[k, l] * ATZ[m, k]
end

function compute_third_term(Z::AbstractArray{T, 3}, S::AbstractArray{T, 3}, ATZ::AbstractArray{T, 3}, i::Int, k::Int, m::Int, l::Int, f::Int) where T
    Z[i, l, f] * S[k, l, f] * ATZ[m, k, f]
end

@doc raw"""
This computes the term 

```math
\sum_j z_{i\ell}s_k(x_{\bullet{}\ell})s_j(x_{\bullet{}\ell})][A^TZ]_{mj}
```
"""
function compute_fourth_term(Z::AbstractMatrix, S::AbstractMatrix, ATZ::AbstractMatrix, i::Int, k::Int, m::Int, l::Int)
    time_steps = size(Z, 2)
    sum(Z[i, l] * S[k, l] * S[j, l] * ATZ[m, j] for j=1:time_steps)
end

function compute_fourth_term(Z::AbstractArray{T, 3}, S::AbstractArray{T, 3}, ATZ::AbstractArray{T, 3}, i::Int, k::Int, m::Int, l::Int, f::Int) where T 
    time_steps = size(Z, 2)
    sum(Z[i, l, f] * S[k, l, f] * S[j, l, f] * ATZ[m, j, f] for j=1:time_steps)
end