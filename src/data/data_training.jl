abstract type AbstractTrainingData end

struct TrainingData{TK <: DataSymbol, TS <: AbstractDataShape, TP <: AbstractProblem, TG <: Dict{Symbol, <:Base.Callable}, TN <: Base.Callable} <: AbstractTrainingData 
    problem::TP
    shape::TS
    get::TG
    symbols::TK
    dim::Int
    noisemaker::TN

    function TrainingData(problem::AbstractProblem, shape::AbstractDataShape, get::NamedTuple, symbols::DataSymbol, dim::Int, noisemaker::Base.Callable)
        new{typeof(symbols),typeof(shape), typeof(problem), typeof(get), typeof(noisemaker)}(problem, shape, get, symbols, dim, noisemaker)
    end

end

function TrainingData(data, get_data::Dict{Symbol, <:Any}, problem = UnknownProblem(); noisemaker =  NothingFunction)
    
    _get_data = copy(get_data)
    
    @assert haskey(get_data, :shape)
    shape = _get_data[:shape](data, _get_data)

    delete!(_get_data, :shape)

    @show get = Dict([(key, (args...)->value(data,args...)) for (key,value) in _get_data])

    symbols = DataSymbol(Tuple(keys(get)))
    
    dim = length(get[1](_index_first(shape)))

    TrainingData(problem, shape, get, symbols, dim, noisemaker)
end

function TrainingData(data::TrainingData; shape = shape(data), get = get(data), symbols = symbols(data), noisemaker =  NothingFunction)
    is_NothingFunction(noisemaker) ? data :  TrainingData(problem(data), shape, get, symbols, dim(data), noisemaker)
end


@inline problem(data::TrainingData) = data.problem
@inline shape(data::TrainingData) = data.shape
@inline get(data::TrainingData) = data.get
@inline data_symbols(data::TrainingData) = data.symbols
@inline symbols(data::TrainingData) = symbols(data_symbols(data))
@inline dim(data::TrainingData) = data.dim
@inline noisemaker(data::TrainingData) = data.noisemaker

@inline get_Δt(data::TrainingData) = get_Δt(data.shape)
@inline get_nb_trajectory(data::TrainingData) = get_nb_trajectory(data.shape)
@inline get_length_trajectory(data::TrainingData, i::Int) = get_length_trajectory(data.shape, i)
@inline get_nb_point(data::TrainingData) = get_nb_point(data.shape)
@inline get_data(data::TrainingData, s::Symbol, args) = data.get[s](args...)

@inline Base.eachindex(data::TrainingData) = eachindex(data.shape)
@inline Base.eachindex(ti::AbstractTrainingIntegrator, data::TrainingData) = eachindex(ti, data.shape)

@inline Base.copy(data::TrainingData) = TrainingData(data)


function reshape_intoSampledData(data::TrainingData)

    new_shape = reshape_intoSampledData!(shape(data))

    #Creating a dictionary and imposing its type
    new_get = Dict(:a,x->x)
    delete!(new_get, :a)

    for s in symbols(data_symbols(data))
        v = []
        for x in eachindex(data)
            push!(v, get_data(data,s, x...))
        end
        new_get[s] = n -> v[n]
    end

    TrainingData(data; shape = new_shape, get = new_get)
end



function reduce_symbols!(data::TrainingData, symbol::DataSymbol)

    #test if it cab be reduced
    can_reduce(data_symbols(data), symbol) ? nothing : throw(ReductionSymbolError(type(data_symbols(data)), type(symbol)))

    #compute the symetric difference of old and new symbols
    toberemoved = symboldiff(data_symbols(data), symbol)

    new_data = TrainingData(data; symbols = symbol)

    #clean get
    clean_get!(new_data, toberemoved)

    new_data
end


function transform_symbols!(data::TrainingData, symbol::DataSymbol)

    #check if it is possible to do the transformation
    

    #compute the symetric difference of old and new symbols
    toberemoved = transform(symbols(data), symbol)



    #clean get
    clean_get!(data, toberemoved)

    TrainingData(data; symbols = symbol)

end


function clean_get!(data::TrainingData, toberemoved::Tuple{Vararg{Symbol}})
    for s in toberemoved
        delete!(get(data), s)
    end
end