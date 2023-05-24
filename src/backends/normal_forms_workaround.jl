@inline function (d::Gradient{false,true})(x::AbstractVecOrMat, ps::Tuple, st::NamedTuple)
    size(x)[1] == d.dim || error("Dimension mismatch.")
    return vcat(x[1:(d.dim÷2)] + ps[3].*d.activation.(x[(d.dim÷2+1):d.dim]),
                    x[(d.dim÷2+1):d.dim])
end

@inline function (d::Gradient{false,false})(x::AbstractVecOrMat, ps::Tuple, st::NamedTuple)
    size(x)[1] == d.dim || error("Dimension mismatch.")
    return vcat(x[1:(d.dim÷2)], x[(d.dim÷2+1):d.dim] + ps[3].*
            d.activation.(x[1:(d.dim÷2)]))
end

@inline function (d::Gradient{true,true})(x::AbstractVecOrMat, ps::Tuple, st::NamedTuple)
    size(x)[1] == d.dim || error("Dimension mismatch.")
    return vcat(x[1:(d.dim÷2)] + ps[1]' * 
                (ps[3] .* d.activation.(ps[1] * x[(d.dim÷2+1):d.dim] .+ vec(ps[2]))), 
                    x[(d.dim÷2+1):d.dim])
end

@inline function(d::Gradient{true,false})(x::AbstractVecOrMat, ps::Tuple, st::NamedTuple)
    size(x)[1] == d.dim || error("Dimension mismatch.")
    return vcat(x[1:(d.dim÷2)], x[(d.dim÷2+1):d.dim] + ps[1]' * 
                    (ps[3] .* d.activation(ps[1]*x[1:(d.dim÷2)] .+ vec(ps[2]))))
end


@inline function (d::Gradient{false,true})(z::Tuple{AbstractFloat, AbstractFloat}, ps::Tuple, st::NamedTuple)
    size(x)[1] == d.dim || error("Dimension mismatch.")
    return vcat(z[1] + ps[3].*d.activation.(z[2]),
                    z[2])
end

@inline function (d::Gradient{false,false})(z::Tuple{AbstractFloat, AbstractFloat}, ps::Tuple, st::NamedTuple)
    size(x)[1] == d.dim || error("Dimension mismatch.")
    return vcat(z[1], z[2] + ps[3].*
            d.activation.(z[1]))
end

@inline function (d::Gradient{true,true})(z::Tuple{AbstractFloat, AbstractFloat}, ps::Tuple, st::NamedTuple)
    size(x)[1] == d.dim || error("Dimension mismatch.")
    return vcat(z[1] + ps[1]' * 
                (ps[3] .* d.activation.(ps[1] * z[2] .+ vec(ps[2]))), 
                    z[2])
end

@inline function(d::Gradient{true,false})(z::Tuple{AbstractFloat, AbstractFloat}, ps::Tuple, st::NamedTuple)
    size(x)[1] == d.dim || error("Dimension mismatch.")
    return vcat(z[1], z[2] + ps[1]' * 
                    (ps[3] .* d.activation(ps[1]*z[1] .+ vec(ps[2]))))
end


