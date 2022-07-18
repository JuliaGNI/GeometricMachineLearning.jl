
function add!(x::AbstractArray, b::AbstractArray)
    @assert shape(x) == shape(b)
    x .+= b
end

function add!(x::AbstractArray, a::AbstractArray, b::AbstractArray)
    @assert shape(x) == shape(a) == shape(b)
    x .= a .+ b
end
