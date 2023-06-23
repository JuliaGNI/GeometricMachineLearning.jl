abstract type AbstractTrainingData end


# Data structure 

struct DataTrajectory <:  AbstractTrainingData
    get_Δt::Base.Callable
    get_nb_trajectory::Base.Callable
    get_length_trajectory::Base.Callable
    get_data::Dict{Symbol, <:Base.Callable}
    
    function DataTrajectory(Data, _get_nb_trajectory::Base.Callable, _get_length_trajectory::Base.Callable,  _get_data::Dict{Symbol, <:Base.Callable}, _get_Δt::Base.Callable = NothingBase.Callable())
        get_Δt() = _get_Δt(Data)
        get_nb_trajectory() = _get_nb_trajectory(Data)
        get_length_trajectory(i) = _get_length_trajectory(Data, i)
        get_data = Dict([(key, (i,n)->value(Data,i,n)) for (key,value) in _get_data])
        new(get_Δt, get_nb_trajectory, get_length_trajectory, get_data)
    end

    function DataTrajectory(Data, _get_data::Dict{Symbol, <:Base.Callable})
        
        @assert haskey(_get_data, :Δt)
        @assert haskey(_get_data, :nb_trajectory)
        @assert haskey(_get_data, :length_trajectory)

        get_Δt() = _get_data[:Δt](Data)
        get_nb_trajectory() = _get_data[:nb_trajectory](Data)
        get_length_trajectory(i) = _get_data[:length_trajectory](Data, i)
       
        get_data = Dict([(key, (i,n)->value(Data,i,n)) for (key,value) in _get_data])

        delete!(get_data, :Δt)
        delete!(get_data, :nb_trajectory)
        delete!(get_data, :length_trajectory)

        new(get_Δt, get_nb_trajectory, get_length_trajectory, get_data)
    end
end


struct DataSampled <:  AbstractTrainingData
    get_nb_point::Base.Callable
    get_data::Dict{Symbol, <:Base.Callable}

    function DataSampled(Data, _get_nb_point::Base.Callable, _get_data::Dict{Symbol, <:Base.Callable})
        get_nb_point() = _get_nb_point(Data)
        get_data = Dict([(key, (i,n)->value(Data,i,n)) for (key,value) in _get_data])
        new(get_nb_point, get_data)
    end
    
    function DataSampled(Data, _get_data::Dict{Symbol, <:Base.Callable})
            
        @assert haskey(_get_data, :nb_points)
        get_nb_point() = _get_data[:nb_points](Data)

        get_data = Dict([(key, n->value(Data,n)) for (key,value) in _get_data])
        
        delete!(get_data, :nb_points)

        new(get_nb_point, get_data)
    end
end


struct DataTarget{T<: AbstractTrainingData} <:  AbstractTrainingData
    data::T
    get_target::Dict{Symbol, <:Base.Callable}

    function DataTarget(data::DataTrajectory, Target, _get_target::Dict{Symbol, <:Base.Callable})
        get_target = Dict([(key, (i,n)->value(Target,i,n)) for (key,value) in _get_target])
        new{typeof(data)}(data, get_target)
    end
    
    function DataTarget(data::DataSampled, Target, _get_target::Dict{Symbol, <:Base.Callable})
        get_target = Dict([(key, n->value(Target,n)) for (key,value) in _get_target])
        new{typeof(data)}(data, get_target)
    end
end


# Useful function
get_Δt(::AbstractTrainingData) = nothing
get_nb_trajectory(::AbstractTrainingData) = nothing 
get_nb_point(::AbstractTrainingData) = nothing
get_length_trajectory(::AbstractTrainingData, args...) = nothing
get_data(::AbstractTrainingData, args...) = nothing
get_target(::AbstractTrainingData, args...) = nothing

get_Δt(data::DataTrajectory) = data.get_Δt()
get_nb_trajectory(data::DataTrajectory) = data.get_nb_trajectory()
get_length_trajectory(data::DataTrajectory, i::Int) = data.get_length_trajectory(i)
get_data(data::DataTrajectory, s::Symbol, i::Int, n::Int) = data.get_data[s](i,n)
get_data(data::DataTrajectory) = data.get_data

get_nb_point(data::DataSampled) = data.get_nb_point()
get_data(data::DataSampled, s::Symbol, n::Int) = data.get_data[s](n)
get_data(data::DataSampled) = data.get_data

get_Δt(data::DataTarget) = get_Δt(data.data)
get_nb_trajectory(data::DataTarget) = get_nb_trajectory(data.data)
get_nb_point(data::DataTarget) = get_nb_point(data.data)
get_length_trajectory(data::DataTarget, i::Int) = get_length_trajectory(data.data, i)
get_data(data::DataTarget, s::Symbol, args...) = get_data(data.data, s, args...)
get_target(data::DataTarget, s::Symbol, args...) = data.get_target[s](args...)
get_data(data::DataTarget) = get_data(data.data)
get_target(data::DataTarget) = data.get_target



# Some assertion Base.Callables to check the type of input data (second argument) against the type of data required (first argument)
test_data_trajectory(::Union{DataSampled, DataTarget{DataSampled}}) = @assert false "Require DataTrajectory!"
test_data_trajectory(::DataTarget{DataTrajectory}) = @warn "Targets are not required!"
test_data_sampled(::DataTarget) = @warn "Targets are not required!"
test_data_target(::Union{DataTrajectory, DataSampled}) = @assert false "Required targets for data!"

test_data_trajectory(::AbstractTrainingData) = nothing
test_data_sampled(::AbstractTrainingData) = nothing
test_data_target(::AbstractTrainingData) = nothing
