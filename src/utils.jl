
@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = tuplejoin(tuplejoin(x, y), z...)

function apply_toNT(ps::NamedTuple, fun_name)
    ps_applied = NamedTuple()
    for key in keys(ps)
        ps_applied = merge(ps_applied, NamedTuple{(key, )}((fun_name(ps[key]), )))
    end
    ps_applied
end

function apply_toNT(ps₁::NamedTuple, ps₂::NamedTuple, fun_name)    
    keys₁ = keys(ps₁)
    @assert keys₁ == keys(ps₂)
    ps_applied = NamedTuple()
    for key in keys(ps₁)
        ps_applied = merge(ps_applied, NamedTuple{(key, )}((fun_name(ps₁[key], ps₂[key]), )))
    end
    ps_applied
end

function apply_toNT(ps₁::NamedTuple, ps₂::NamedTuple, ps₃::NamedTuple, fun_name)    
    keys₁ = keys(ps₁)
    @assert keys₁ == keys(ps₂) == keys(ps₃)
    ps_applied = NamedTuple()
    for key in keys(ps₁)
        ps_applied = merge(ps_applied, NamedTuple{(key, )}((fun_name(ps₁[key], ps₂[key], ps₃[key]), )))
    end
    ps_applied
end

#overloaded + operation to work with NamedTuples
_add(dx₁::NamedTuple, dx₂::NamedTuple) = apply_toNT(dx₁, dx₂, _add)
_add(A::AbstractArray, B::AbstractArray) = A + B 


#second argumen pl is "patch length"
#this splits the image into patches of size pl×pl and then arranges them into a matrix,
#the columns of the matrix give the patch number.

function flatten(image_patch::AbstractMatrix)
    n, m = size(image_patch)
    reshape(image_patch, n*m)
end

function split_and_flatten(image::AbstractMatrix, pl)
    n, m = size(image)
    @assert n == m
    @assert n%pl == 0
    #square root of patch number
    pnsq = n ÷ pl
    hcat(Tuple(vcat(map(j -> map(i -> flatten(image[pl*(i-1)+1:pl*i,pl*(j-1)+1:pl*j,1]), 1:pnsq),1:pnsq)...))...)
end

function add!(C::AbstractVecOrMat, A::AbstractVecOrMat, B::AbstractVecOrMat)
    @assert size(A) == size(B) == size(C)
    C .= A + B
end
