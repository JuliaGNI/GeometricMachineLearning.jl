
abstract type AbstractOptimizer end

function apply!(o::AbstractOptimizer, state, model::Lux.Chain, x, dx)
    for i in 1:length(model)
    #for i in eachindex(model, x, dx)
        update_layer!(o, state, model[i], x[i], dx[i])
    end
end
