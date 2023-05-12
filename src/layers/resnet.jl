struct ResNet{use_bias, F1, F2, F3} <: Lux.AbstractExplicitLayer
    activation::F1
    dim::Int
    init_weight::F2
    init_bias::F3
end

function Base.show(io::IO, d::ResNet{use_bias}) where {use_bias}
    print(io, "ResNet($(d.dim) => $(d.dim)")
    (d.activation == Lux.identity) || print(io, ", $(d.activation)")
    use_bias || print(io, ", bias=false")
    return print(io, ")")
end

function ResNet(mapping::Pair{<:Int, <:Int}, activation=Lux.identity; kwargs...)
    return ResNet(first(mapping), last(mapping), activation; kwargs...)
end

function ResNet(dim::Int, activation=identity; init_weight=Lux.glorot_uniform,
               init_bias=Lux.zeros32, use_bias::Bool=true, bias::Union{Missing, Bool}=missing,
               allow_fast_activation::Bool=true)
    activation = allow_fast_activation ? NNlib.fast_act(activation) : activation

    # Deprecated Functionality (Remove in v0.5)
    if !ismissing(bias)
        Base.depwarn("`bias` argument to `ResNet` has been deprecated and will be removed" *
                     " in v0.5. Use `use_bias` kwarg instead.", :ResNet)
        if !use_bias
            throw(ArgumentError("Both `bias` and `use_bias` are set. Please only use " *
                                "the `use_bias` keyword argument."))
        end
        use_bias = bias
    end

    dtype = (use_bias, typeof(activation), typeof(init_weight), typeof(init_bias))
    return ResNet{dtype...}(activation, dim, init_weight, init_bias)
end

function Lux.initialparameters(rng::Random.AbstractRNG, d::ResNet{use_bias}) where {use_bias}
    if use_bias
        return (weight=d.init_weight(rng, d.dim, d.dim),
                bias=d.init_bias(rng, d.dim, 1))
    else
        return (weight=d.init_weight(rng, d.dim, d.dim),)
    end
end

function Lux.parameterlength(d::ResNet{use_bias}) where {use_bias}
    return use_bias ? d.dim * (d.dim + 1) : d.dim * d.dim
end
Lux.statelength(d::ResNet) = 0

@inline function Lux.apply(d::ResNet{false}, x::AbstractVecOrMat, ps, st::NamedTuple)
    return x + Lux.__apply_activation(d.activation, ps.weight * x), st
end

@inline function Lux.apply(d::ResNet{false}, x::AbstractArray, ps, st::NamedTuple)
    sz = size(x)
    x_reshaped = reshape(x, sz[1], :)
    return x + reshape(Lux.__apply_activation(d.activation, ps.weight * x_reshaped), d.dim,
                   sz[2:end]...), st
end

@inline function Lux.apply(d::ResNet{true}, x::AbstractVector, ps, st::NamedTuple)
    return x + Lux.__apply_activation(d.activation, ps.weight * x .+ vec(ps.bias)), st
end

@inline function Lux.apply(d::ResNet{true}, x::AbstractMatrix, ps, st::NamedTuple)
    return x + Lux.__apply_activation(d.activation, ps.weight * x .+ ps.bias), st
end

@inline function Lux.apply(d::ResNet{true}, x::AbstractArray, ps, st::NamedTuple)
    sz = size(x)
    x_reshaped = reshape(x, sz[1], :)
    return x + (reshape(Lux.__apply_activation(d.activation, ps.weight * x_reshaped .+ ps.bias),
                    d.dim, sz[2:end]...), st)
end