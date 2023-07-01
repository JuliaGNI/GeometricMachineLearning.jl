pretransform(::AbstractTrainingIntegrator, params::NamedTuple) = params, nothing

posttransform(::AbstractTrainingIntegrator, params,  args...) = params


TuppleNeededTrainingIntegrator = Union{HnnTrainingIntegrator, LnnTrainingIntegrator}

function pretransform(::TuppleNeededTrainingIntegrator, params::NamedTuple)

    # convert parameters to tuple
    params_tuple = Tuple([Tuple(x) for x in params])

    keys_1 = keys(params)
    keys_2 = [keys(x) for x in values(params)]

    params_tuple, (keys_1, keys_2)
end


function posttransform(::HnnTrainingIntegrator, params_grad::Tuple, keys)

    NamedTuple(zip(keys[1],[NamedTuple(zip(k,x)) for (k,x) in zip(keys[2],params_grad)]))

end