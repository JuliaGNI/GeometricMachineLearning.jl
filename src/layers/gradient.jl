import Lux

#=
If full_grad is true, then the Gradient layer also has a bias
=#

#activation layer
struct Gradient{full_grad, change_q, F1, F2, F3, F4} <: Lux.AbstractExplicitLayer
        activation::F1
        dim::Int
        dim2::Int
        init_weight::F2
        init_bias::F3
        init_scale::F4
end

#check: input is even; make dim2 an optional argument for full_grad=false
function Gradient(dim::Int, dim2::Int=dim, activation=identity; init_weight=Lux.glorot_uniform,
                init_bias=Lux.zeros32, init_scale=Lux.glorot_uniform, full_grad::Bool=true,
                change_q::Bool=true, allow_fast_activation::Bool=true)
        activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
        iseven(dim) && iseven(dim2) || error("Dimensions must be even!")
        dim2 ≥ dim || error("Second dimension should be bigger than the first!")
        dtype = (full_grad, change_q, typeof(activation), typeof(init_weight), typeof(init_bias), 
                 typeof(init_scale))
        return Gradient{dtype...}(activation, dim, dim2, init_weight, init_bias, init_scale)
end


function initialparameters(backend::Backend, ::Type{T}, d::Gradient{full_grad}; rng::AbstractRNG = Random.default_rng()) where {full_grad, T}
        if full_grad
                return (weight=d.init_weight(rng, d.dim2÷2, d.dim÷2),
                        bias=d.init_bias(rng, d.dim2÷2, 1),
                        scale=d.init_scale(rng,d.dim2÷2,1))
        else
                return (scale=d.init_scale(rng, d.dim÷2, 1),)
        end
end


function parameterlength(d::Gradient{full_grad}) where {full_grad}
        return full_grad ? d.dim2÷2 * (d.dim÷2 + 2) : d.dim÷2
end


@inline function (d::Gradient{false,true})(x::AbstractVecOrMat, ps::NamedTuple)
        size(x)[1] == d.dim || error("Dimension mismatch.")
        return vcat(x[1:(d.dim÷2)] + ps.scale.*d.activation.(x[(d.dim÷2+1):d.dim]),
                        x[(d.dim÷2+1):d.dim])
end

@inline function (d::Gradient{false,false})(x::AbstractVecOrMat, ps::NamedTuple)
        size(x)[1] == d.dim || error("Dimension mismatch.")
        return vcat(x[1:(d.dim÷2)], x[(d.dim÷2+1):d.dim] + ps.scale.*
                d.activation.(x[1:(d.dim÷2)]))
end

@inline function (d::Gradient{true,true})(x::AbstractVecOrMat, ps::NamedTuple)
        size(x)[1] == d.dim || error("Dimension mismatch.")
        return vcat(x[1:(d.dim÷2)] + ps.weight' * 
                    (ps.scale .* d.activation.(ps.weight * x[(d.dim÷2+1):d.dim] .+ vec(ps.bias))), 
                        x[(d.dim÷2+1):d.dim])
end

@inline function(d::Gradient{true,false})(x::AbstractVecOrMat, ps::NamedTuple)
        size(x)[1] == d.dim || error("Dimension mismatch.")
        return vcat(x[1:(d.dim÷2)], x[(d.dim÷2+1):d.dim] + ps.weight' * 
                        (ps.scale .* d.activation(ps.weight*x[1:(d.dim÷2)] .+ vec(ps.bias))))
end

#######
#this is for GPU support (doesn't support indexing arrays); for now only CUDA!!

function assign_first_half!(q, x)
        i = CUDA.threadIdx().x
        q[i] = x[i]
        return 
end

function assign_second_half!(p, x, N)
        i = CUDA.threadIdx().x
        p[i] = x[i+N]
        return 
end

function assign_q_and_p(x, N)
        q = CUDA.zeros(eltype(x), N)
        p = CUDA.zeros(eltype(x), N)
        CUDA.@cuda threads=N assign_first_half!(q, x)
        CUDA.@cuda threads=N assign_second_half!(p, x, N)
        q, p
end

@inline function (d::Gradient{false,true})(x::AbstractGPUVecOrMat, ps)
        size(x)[1] == d.dim || error("Dimension mismatch.")
        N = d.dim÷2
        q, p = assign_q_and_p(x, N)
        return vcat(q + ps.scale.*d.activation.(p), p)
end

@inline function (d::Gradient{false,false})(x::AbstractGPUVecOrMat, ps)
        size(x)[1] == d.dim || error("Dimension mismatch.")
        N = d.dim÷2 
        q, p = assign_q_and_p(x, N)
        return vcat(q, p + ps.scale.*d.activation.(q))
end

@inline function (d::Gradient{true,true})(x::AbstractGPUVecOrMat, ps)
        size(x)[1] == d.dim || error("Dimension mismatch.")
        N = d.dim÷2 
        q, p = assign_q_and_p(x, N)
        return vcat(q + ps.weight' * 
                (ps.scale .* d.activation.(ps.weight * p .+ vec(ps.bias))), p)
end

@inline function(d::Gradient{true,false})(x::AbstractGPUVecOrMat, ps)
        size(x)[1] == d.dim || error("Dimension mismatch.")
        N = d.dim÷2 
        q, p = assign_q_and_p(x, N)
        return vcat(q, p + ps.weight' * 
                        (ps.scale .* d.activation(ps.weight*q .+ vec(ps.bias))))
end

@inline function (d::Gradient{false,true})(x::AbstractVecOrMat, ps::Tuple)
        size(x)[1] == d.dim || error("Dimension mismatch.")
        return vcat(x[1:(d.dim÷2)] + ps[3].*d.activation.(x[(d.dim÷2+1):d.dim]),
                        x[(d.dim÷2+1):d.dim])
end

@inline function (d::Gradient{false,false})(x::AbstractVecOrMat, ps::Tuple)
        size(x)[1] == d.dim || error("Dimension mismatch.")
        return vcat(x[1:(d.dim÷2)], x[(d.dim÷2+1):d.dim] + ps[3].*
                d.activation.(x[1:(d.dim÷2)]))
end

@inline function (d::Gradient{true,true})(x::AbstractVecOrMat, ps::Tuple)
        size(x)[1] == d.dim || error("Dimension mismatch.")
        return vcat(x[1:(d.dim÷2)] + ps[1]' * 
                    (ps[3] .* d.activation.(ps[1] * x[(d.dim÷2+1):d.dim] .+ vec(ps[2]))), 
                        x[(d.dim÷2+1):d.dim])
end

@inline function(d::Gradient{true,false})(x::AbstractVecOrMat, ps::Tuple)
        size(x)[1] == d.dim || error("Dimension mismatch.")
        return vcat(x[1:(d.dim÷2)], x[(d.dim÷2+1):d.dim] + ps[1]' * 
                        (ps[3] .* d.activation(ps[1]*x[1:(d.dim÷2)] .+ vec(ps[2]))))
end

