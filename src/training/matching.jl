
#=
    The matching function checks the correspondence between the training method, the form of the data and the symbols in the data. 
        - It first checks the correspondence between the shape of the data and that required by the method. If sample data is required but trajectory data is provided,
           the shape of the training data is converted into the correct shape.
        - It then checks that the symbols in the data match those required by the method. If the symbols don't match, we first try a reduction, then we try a transformation to convert the data.
=#

function matching(ti::TrainingIntegrator, data::AbstractTrainingData)

    # matching between the shape required by the method and the one given
    if shape(ti) != typeof(shape(data))
        if shape(ti) == SampledData
            reshape_intoSampledData!(data)
        else 
            @assert shape(ti) == typeof(shape(data))
        end
    end
    # matching between the symbols required by the method and those given
    if symbols(ti) != type(data_symbols(data))
        try reduce_symbols!(data, DataSymbol{symbols(ti)}())
        catch
            try Transform_symbols!(data, DataSymbol{symbols(ti)}()) 
                println("Automatic transformatiom ")
            catch 
                @assert symbols(ti) == type(data_symbols(data))
            end
        end
    end
end


