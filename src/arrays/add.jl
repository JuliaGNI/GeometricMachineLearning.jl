
function add!(x::AbstractArray, b::AbstractArray)
    @assert axes(x) == axes(b)
    x .+= b
end

function add!(x::AbstractArray, a::AbstractArray, b::AbstractArray)
    @assert axes(x) == axes(a) == axes(b)
    x .= a .+ b
end
