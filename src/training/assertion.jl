# Assertion for good usage of training integrator

required_key(ti::AbstractTrainingIntegrator) = @warn "No recquired_key functions for "*string(typeof(ti))*"!"

data_goal(ti::AbstractTrainingIntegrator) = @warn "No data recquirement for "*string(typeof(ti))*". Errors may occur."; nothing

function assert(ti::AbstractTrainingIntegrator, data::AbstractTrainingData)
    for type_data in data_goal(ti)
        type_data(data)
    end
    for key in required_key(ti)
        @assert (haskey(get_data(data), key) || haskey(get_target(data), key)) "You forgot the key "*string(key)*"!"
    end

end
