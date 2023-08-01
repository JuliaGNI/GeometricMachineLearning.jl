
@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = tuplejoin(tuplejoin(x, y), z...)


function apply_toNT(fun, ps::NamedTuple...)
    for p in ps
        @assert keys(ps[1]) == keys(p)
    end
    NamedTuple{keys(ps[1])}(fun(p...) for p in zip(ps...))
end

# overloaded + operation to work with NamedTuples
_add(dx₁::NamedTuple, dx₂::NamedTuple) = apply_toNT( _add, dx₁, dx₂)
_add(A::AbstractArray, B::AbstractArray) = A + B 


# overloaded similar operation to work with NamedTuples
_similar(x) = similar(x)

function _similar(x::Tuple)
    Tuple(_similar(_x) for _x in x)
end

function _similar(x::NamedTuple)
    NamedTuple{keys(x)}(_similar(values(x)))
end


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

struct NothingFunction <: Function end
(::NothingFunction)(args...) = nothing
is_NothingFunction(f::Function) = typeof(f)==NothingFunction


function Base.:+(a::Float64, b::Tuple{Float64})
    x, = b
    return a+x
end

function Base.:+(a::Vector{Float64}, b::Tuple{Float64})
    x, = b
    y, = a
    return y+x
end

#Zygote.OneElement(t1::Tuple{Float64}, t2::Tuple{Int64}, t3::Tuple{Base.OneTo{Int64}}) = Zygote.OneElement(t1[1], t2, t3)

function type_without_brace(var)
    type_str = string(typeof(var))
    replace(type_str, r"\{.*\}"=>"")
end


function add!(dx₁::NamedTuple, dx₂::NamedTuple, dx₃::NamedTuple)
    apply_toNT(add!, dx₁, dx₂, dx₃)
end

struct UnknownProblem <: AbstractProblem end

_tuplediff(t₁::Tuple,t₂::Tuple) = tuple(setdiff(Set(t₁),Set(t₂))...)

@inline next(i::Int,j::Int) = (i,j+1)
@inline next(i::Int) = (i+1,)

function center_align_text(text,width)
    padding = max(0, width - length(text))
    left_padding = repeat(" ",padding ÷2)
    right_padding = repeat(" ", padding - length(left_padding))
    aligned_text = left_padding * text * right_padding
    return aligned_text
end

const ∞ = Inf

#The following are fallback functions - maybe you want to put them into a separate file

function global_section(::AbstractVecOrMat)
    nothing
end


struct CPUDevice end 

const Device = Union{CUDA.CuDevice, CPUDevice}

function convert_to_dev(::CUDA.CuDevice, A::AbstractArray)
    CUDA.cu(A)
end

function convert_to_dev(::CPUDevice, A::AbstractVector)
    Vector(A)
end

function convert_to_dev(::CPUDevice, A::AbstractMatrix)
    Matrix(A)
end

#=
function Lux.setup(dev::Device, rng::Random.AbstractRNG, d::Lux.AbstractExplicitLayer)
    map_to_dev(A::AbstractArray) = convert_to_dev(dev, A)
    map_to_dev(ps::NamedTuple) = apply_toNT(ps, map_to_dev)
    ps, st = Lux.setup(rng, d) 
    ps = map_to_dev(ps)
    ps, st
end
=#
