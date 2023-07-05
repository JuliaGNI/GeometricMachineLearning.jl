
#=
    The Assert function checks the correspondence between the training method, the form of the data and the symbols in the data. 
        - It first checks the correspondence between the shape of the data and that required by the method. If sample data is required but trajectory data is provided,
           the shape of the training data is converted into the correct shape.
        - It then checks that the symbols in the data match those required by the method. If the symbols don't match, we first try a reduction, then we try a transformation to convert the data.
=#

function assert(ti::AbstractTrainingIntegrator, data::AbstractTrainingData)

    if shape(ti) != typeof(shape(data))

        if shape(ti) == SampledData

            reshape_intoSampledData!(data)

        else 
            @assert shape(ti) == typeof(shape(data))
        end
    end

    if symbols(ti) != type(symbols(data))

        if can_reduce(symbols(data), DataSymbol{symbols(ti)}())

            reduce_symbols!(data, DataSymbol{symbols(ti)}())

        elseif can_transform(symbols(data), symbols(ti))

            transform_symbols!(data, DataSymbol{symbols(ti)}())

        else
            @assert symbols(ti) == type(symbols(data))
        end
    end
end