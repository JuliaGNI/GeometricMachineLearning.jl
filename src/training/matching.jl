
#=
    The matching function checks the correspondence between the training method, the form of the data and the symbols in the data. 
        - It first checks the correspondence between the shape of the data and that required by the method. If sample data is required but trajectory data is provided,
           the shape of the training data is converted into the correct shape.
        - It then checks that the symbols in the data match those required by the method. If the symbols don't match, we first try a reduction, then we try a transformation to convert the data.
=#

function matching(ti::TrainingMethod, data::AbstractTrainingData)

    new_data = data

    # matching between the shape required by the method and the one given
    if shape(ti) != typeof(shape(new_data))
        if shape(ti) == SampledData
            new_data = reshape_intoSampledData(new_data)
        else 
            @assert shape(ti) == typeof(shape(new_data))
        end
    end
    # matching between the symbols required by the method and those given
    if symbols(ti) != type(data_symbols(new_data))
        try new_data = reduce_symbols(new_data, DataSymbol{symbols(ti)}())
        catch
            try new_data = Transform_symbols(new_data, DataSymbol{symbols(ti)}()) 
                println("Automatic transformatiom ")
            catch 
                @assert symbols(ti) == type(data_symbols(new_data))
            end
        end
    end
    new_data
end


