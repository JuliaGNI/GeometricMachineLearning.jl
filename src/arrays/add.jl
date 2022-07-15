
function add!(x::AbstractArray, b::AbstractArray)
    @assert shape(x) == shape(b)
    x .+= b
end
