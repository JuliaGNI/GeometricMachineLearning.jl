abstract type AbstractTrainingData end


# Data structure 

struct DataTrajectory <:  AbstractTrainingData
    get_Δt::Function
    get_nb_trajectory::Function
    get_length_trajectory::Function
    get_data::Dict{Symbol, <:Function}
    
    function DataTrajectory(Data, _get_nb_trajectory::Function, _get_length_trajectory::Function,  _get_data::Dict{Symbol, <:Function}, _get_Δt::Function = NothingFunction())
        get_Δt() = _get_Δt(Data)
        get_nb_trajectory() = _get_nb_trajectory(Data)
        get_length_trajectory(i) = _get_length_trajectory(Data, i)
        get_data = Dict([(key, (i,n)->value(Data,i,n)) for (key,value) in _get_data])
        new(get_Δt, get_nb_trajectory, get_length_trajectory, get_data)
    end

    function DataTrajectory(Data, _get_data::Dict{Symbol, <:Function})
        
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
    get_nb_point::Function
    get_data::Dict{Symbol, <:Function}

    function DataSampled(Data, _get_nb_point::Function, _get_data::Dict{Symbol, <:Function})
        get_nb_point() = _get_nb_point(Data)
        get_data = Dict([(key, (i,n)->value(Data,i,n)) for (key,value) in _get_data])
        new(get_nb_point, get_data)
    end
    
    function DataSampled(Data, _get_data::Dict{Symbol, <:Function})
            
        @assert haskey(_get_data, :nb_points)
        get_nb_point() = _get_data[:nb_points](Data)

        get_data = Dict([(key, n->value(Data,n)) for (key,value) in _get_data])
        
        delete!(get_data, :nb_points)

        new(get_nb_point, get_data)
    end
end


struct DataTarget{T<: AbstractTrainingData} <:  AbstractTrainingData
    data::T
    get_data::Dict{Symbol, <:Function}
    get_target::Dict{Symbol, <:Function}

    function DataTarget(data::DataTrajectory, Target, _get_target::Dict{Symbol, <:Function})
        get_target = Dict([(key, (i,n)->value(Target,i,n)) for (key,value) in _get_target])
        new{typeof(data)}(data, data.get_data, get_target)
    end
    
    function DataTarget(data::DataSampled, Target, _get_target::Dict{Symbol, <:Function})
        get_target = Dict([(key, n->value(Target,n)) for (key,value) in _get_target])
        new{typeof(data)}(data, data.get_data, get_target)
    end
end


# Useful Functions
#get_Δt(::AbstractTrainingData) = @error "You are using an undefined  "
#get_Δt(::DataTrajectory)
#get_Δt(::DataTrajectory)



# Some assertion functions to check the type of input data (second argument) against the type of data required (first argument)
assert(::DataTrajectory, ::Union{DataSampled, DataTarget{DataSampled}}) = @assert false "Require DataTrajectory!"
assert(::DataTarget{DataTrajectory}, ::Union{DataSampled, DataTarget{DataSampled}}) = @warn "Targets are not required!"
assert(::DataTarget, ::Union{DataTrajectory, DataSampled}) = @assert false "Required targets for data!"
assert(::AbstractTrainingData, ::Nothing) = nothing
