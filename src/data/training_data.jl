abstract type AbstractTrainingData end


struct TrainingData{TK <: AbstractGiven, TS <: AbstractDataShape, TP <: AbstractProblem, TG <: NamedTuple, TN <: Base.Callable} <: AbstractTrainingData 
    problem::TP
    shape::TS
    get::TG
    keys::TK
    dim::Int
    noisemaker::TN

    function TrainingData(problem::AbstractProblem, shape::AbstractDataShape, get::NamedTuple, keys::AbstractGiven, dim::Int, noisemaker::Base.Callable)
        new{typeof(keys),typeof(shape), typeof(problem), typeof(get), typeof(noisemaker)}(problem, shape, get, keys, dim, noisemaker)
    end

end

function TrainingData(data, _get_data::Dict{Symbol, <:Base.Callable}, problem = UnknownProblem; noisemaker =  NothingFunction)
        
    @assert haskey(_get_data, :shape)
    shape = _get_data[:shape](data, _get_data)

    delete!(_get_data, :shape)

    get = NamedTuple([(key, (args...)->value(Data,args...)) for (key,value) in _get_data])

    #keys ?
    
    dim = length(get[Tuple(keys(get))[1]])

    TrainingData(problem, shape, get, keys, dim, noisemaker)
end

function TrainingData(data::TrainingData; noisemaker =  NothingFunction)
    is_NothingFunction(noisemaker) ? data :  TrainingData(problem(data), shape(data), get(data), keys(data), dim(data), noisemaker)
end


@inline problem(data::TrainingData) = data.problem
@inline shape(data::TrainingData) = data.shape
@inline get(data::TrainingData) = data.get
@inline keys(data::TrainingData) = data.keys
@inline dim(data::TrainingData) = data.dim
@inline noisemaker(data::TrainingData) = data.noisemaker

@inline get_Δt(data::TrainingData) = get_Δt(data.shape)
@inline get_nb_trajectory(data::TrainingData) = get_nb_trajectory(data.shape)
@inline get_length_trajectory(data::TrainingData, i::Int) = get_length_trajectory(data.shape, i)
@inline get_data(data::TrainingData, s::Symbol, args) = data.get[s](args...)

