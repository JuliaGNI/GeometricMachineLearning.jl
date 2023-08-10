pretransform(::AbstractTrainingMethod, params::NamedTuple) = params, nothing

posttransform(::AbstractTrainingMethod, params,  args...) = params


TuppleNeededTrainingMethod = Union{HnnTrainingMethod, LnnTrainingMethod}

function pretransform(::TuppleNeededTrainingMethod, params::NamedTuple)

    # convert parameters to tuple
    params_tuple = Tuple([Tuple(x) for x in params])

    keys_1 = keys(params)
    keys_2 = [keys(x) for x in values(params)]

    params_tuple, (keys_1, keys_2)
end


function posttransform(::TuppleNeededTrainingMethod, params_grad::Tuple, keys)

    NamedTuple(zip(keys[1],[NamedTuple(zip(k,x)) for (k,x) in zip(keys[2],params_grad)]))

end