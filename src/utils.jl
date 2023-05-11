
@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = tuplejoin(tuplejoin(x, y), z...)

#overloaded + operation to work with NamedTuples
function Base.:+(dx₁::NamedTuple, dx₂::NamedTuple)
    keys₁ = keys(dx₁)
    @assert keys₁ == keys(dx₂)
    dx_sum = NamedTuple()
    for key in keys₁
        dx_sum = merge(dx_sum, NamedTuple{(key,)}((dx₁[key] + dx₂[key],)))
    end
    dx_sum
end
