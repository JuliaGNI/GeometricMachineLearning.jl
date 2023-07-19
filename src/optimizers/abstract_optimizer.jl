abstract type AbstractOptimizer end


function optimization_step!(o::AbstractOptimizer, d::Lux.AbstractExplicitLayer, ps::NamedTuple, C::NamedTuple, dx::NamedTuple)
    gx = rgrad(ps, dx)
    λY = GlobalSection(ps)
    B = global_rep(λY, gx)
    update!(o, C, B)
    ps₂ = retraction(d, B)
    apply_section!(ps, λY, ps₂)
end

#add a routine that can deal with a single layer (no Lux.Chain)
#function optimization_step!(o::AbstractOptimizer, model::Lux.AbstractExplicitLayer, ps::NamedTuple, cache::NamedTuple, dx::NamedTuple)
#end

function optimization_step!(o::AbstractOptimizer, model::Lux.Chain, ps::NamedTuple, loss)
    dx = Zygote.gradient(ps -> loss(ps), ps)[1]
    optimization_step!(o, model, ps, dx)
end 

rgrad(ps::NamedTuple, dx::NamedTuple) = apply_toNT(ps, dx, rgrad)

function rgrad(Y::AbstractVecOrMat, dx::AbstractVecOrMat)
    @assert size(Y) == size(dx)
    dx
end

function apply_toNT(o::AbstractOptimizer, ps₁::NamedTuple, ps₂::NamedTuple, fun)
    apply_toNT(ps₁, ps₂, (ps₁, ps₂) -> fun(o, ps₁, ps₂))
end

function update!(o::AbstractOptimizer, C::NamedTuple, B::NamedTuple)
    apply_toNT(o, C, B, update!)
end
