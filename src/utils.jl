# Convenient structure
struct NothingFunction <: Function end
(::NothingFunction)(args...) = nothing
is_NothingFunction(f::Function) = typeof(f)==NothingFunction

struct UnknownProblem <: AbstractProblem end

const ∞ = Inf

# Functions on typple and named tuple

@inline next(i::Int,j::Int) = (i,j+1)
@inline next(i::Int) = (i+1,)

@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = tuplejoin(tuplejoin(x, y), z...)


rdevelop(x) = x
rdevelop(t::Tuple{Any}) = [rdevelop(t[1])...]
rdevelop(t::Tuple) = [rdevelop(t[1])..., rdevelop(t[2:end])...]
rdevelop(t::NamedTuple) = vcat([[rdevelop(e)...] for e in t]...)

develop(x) = [x]
develop(t::Tuple{Any}) = [develop(t[1])...]
develop(t::Tuple) = [develop(t[1])..., develop(t[2:end])...]
develop(t::NamedTuple) = vcat([[develop(e)...] for e in t]...)


_tuplediff(t₁::Tuple,t₂::Tuple) = tuple(setdiff(Set(t₁),Set(t₂))...)

function apply_toNT(fun, ps::NamedTuple...)
    for p in ps
        @assert keys(ps[1]) == keys(p)
    end
    NamedTuple{keys(ps[1])}(fun(p...) for p in zip(ps...))
end

# overloaded + operation 
_add(dx₁::NamedTuple, dx₂::NamedTuple) = apply_toNT( _add, dx₁, dx₂)
_add(A::AbstractArray, B::AbstractArray) = A + B 

function add!(C::AbstractVecOrMat, A::AbstractVecOrMat, B::AbstractVecOrMat)
    @assert size(A) == size(B) == size(C)
    C .= A + B
end

function add!(dx₁::NamedTuple, dx₂::NamedTuple, dx₃::NamedTuple)
    apply_toNT(add!, dx₁, dx₂, dx₃)
end

function Base.:+(a::Float64, b::Tuple{Float64})
    x, = b
    return a+x
end

function Base.:+(a::Vector{Float64}, b::Tuple{Float64})
    x, = b
    y, = a
    return y+x
end


# overloaded similar operation to work with NamedTuples
_similar(x) = similar(x)

function _similar(x::Tuple)
    Tuple(_similar(_x) for _x in x)
end

function _similar(x::NamedTuple)
    NamedTuple{keys(x)}(_similar(values(x)))
end

# utils functions on string

function type_without_brace(var)
    type_str = string(typeof(var))
    replace(type_str, r"\{.*\}"=>"")
end

function center_align_text(text,width)
    padding = max(0, width - length(text))
    left_padding = repeat(" ",padding ÷2)
    right_padding = repeat(" ", padding - length(left_padding))
    aligned_text = left_padding * text * right_padding
    return aligned_text
end


#The following are fallback functions - maybe you want to put them into a separate file

function global_section(::AbstractVecOrMat)
    nothing
end