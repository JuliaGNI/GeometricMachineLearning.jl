
abstract type AbstractOptimizer end 

function apply!(o::AbstractOptimizer, state, model, x, dx)
    for i in eachindex(model)
        update_layer!(o, state, model[i], x[i], dx[i])
    end
end
