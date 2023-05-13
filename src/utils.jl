
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
    for key in keys(ps)
        ps_applied = merge(ps_applied, NamedTuple{(key, )}((fun_name(ps₁[key], ps₂[key]), )))
    end
    ps_applied
end

#overloaded + operation to work with NamedTuples
_add(dx₁::NamedTuple, dx₂::NamedTuple) = apply_toNT(dx₁, dx₂, _add)
_add(A::AbstractArray, B::AbstractArray) = A + B 

