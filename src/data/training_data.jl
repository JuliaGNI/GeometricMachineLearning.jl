abstract type AbstractTrainingData end

struct TrainingData{TK <: AbstractGiven, TS <: AbstractDataShape, TP <: AbstractProblem, TG <: NamedTuple, TN <: Base.Callable} <: AbstractTrainingData 
    problem::TP
    shape::TS
    get::TG
    keys::TK
    dim::Int
    noisemaker::TN

    function TrainingData(data, _get_data::Dict{Symbol, <:Base.Callable}, problem = UnknownProblem; noisemaker =  NothingFunction)
        
        @assert haskey(_get_data, :shape)
        shape = _get_data[:shape](data, _get_data)

        delete!(_get_data, :shape)

        get = NamedTuple([(key, (args...)->value(Data,args...)) for (key,value) in _get_data])

        #keys ?
        
        dim = length(get[Tuple(keys(get))[1]])

        new{typeof(keys),typeof(shape), typeof(problem), typeof(get), typeof(noisemaker)}(problem, shape, get, keys, dim, noisemaker)
    end

end


