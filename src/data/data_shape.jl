#=
    AbstractDataShape contains the shape of data which can be :
        - a TrajectoryData,
        - a SampledData.
    This structure is helful to know how are organised the data.
=#

abstract type AbstractDataShape end

#=
    TrajectoryData is a shape of data which represents severals trajectories, i.e a set of sequences of point separated by a timestep. 
    It contains
        - Δt : the time step,
        - nb_trajectory : the number of trajectory,
        - length_trajectory : an array which contains the length of each trajectories.
=#

struct TrajectoryData <:  AbstractDataShape
    Δt::Union{Real,Nothing}
    nb_trajectory::Int
    length_trajectory::AbstractArray{Int}
    
    function TrajectoryData(Data, _get_data::Dict{Symbol, <:Any})
        
        @assert haskey(_get_data, :nb_trajectory)
        @assert haskey(_get_data, :length_trajectory)

        Δt = haskey(_get_data, :Δt) ? _get_data[:Δt](Data) : nothing

        nb_trajectory = _get_data[:nb_trajectory](Data)
        length_trajectory = [_get_data[:length_trajectory](Data, i) for i in 1:nb_trajectory]

        delete!(_get_data, :Δt)
        delete!(_get_data, :nb_trajectory)
        delete!(_get_data, :length_trajectory)
       
        new(Δt, nb_trajectory, length_trajectory)
    end
end

#=
   SampledData is a shape of data which represents data without any kind of link between them. 
    It contains just
        - nb_point : the size of data.
=#

struct SampledData <: AbstractDataShape
    nb_point::Int

    function SampledData(_nb_point::Int) 
        new(_nb_point)
    end
  
    function SampledData(Data, _get_data::Dict{Symbol, <:Any})
            
        @assert haskey(_get_data, :nb_points)
        nb_point = _get_data[:nb_points](Data)

        delete!(_get_data, :nb_points)

        new(nb_point)
    end
end

@inline get_Δt(::AbstractDataShape) = nothing
@inline get_nb_trajectory(::AbstractDataShape) = nothing 
@inline get_nb_point(::AbstractDataShape) = nothing
@inline get_length_trajectory(::AbstractDataShape, args...) = nothing
@inline get_data(::AbstractDataShape, args...) = nothing

@inline get_Δt(data::TrajectoryData) = data.Δt
@inline get_nb_trajectory(data::TrajectoryData) = data.nb_trajectory
@inline get_length_trajectory(data::TrajectoryData, i::Int) = data.length_trajectory[i]
@inline get_nb_point(data::SampledData) = data.nb_point

@inline next(i,j) = (i,j+1)
@inline next(i) = (i+1,)

@inline _index_first(:AbstractDataShape) = nothing
@inline _index_first(:TrajectoryData) = (1,1)
@inline _index_first(:SampledData) = 1

@inline eachindex(::AbstractDataShape) = nothing
@inline eachindex(data::TrajectoryData) = vcat([[(i,j) for j in  get_length_trajectory(data,i)] for i in get_nb_trajectory(data)]...)
@inline eachindex(data::SampledData) = 1:get_nb_point(data)

reshape_intoSampledData!(data::AbstractDataShape) = @error "It is not possible to convert "*string(typeof(data))*" into SampledData."
reshape_intoSampledData!(data::SampledData) = data
reshape_intoSampledData!(data::TrajectoryData) = SampledData(sum([get_length_trajectory(data,i) for i in 1:get_nb_trajectory(data)]...))

iterate(data::AbstractDataShape, state = 1, args...) = nothing 
iterate(data::SampledData, state = 1, s::Symbol) = state > get_nb_point(data) ? nothing : ()
iterate(data::TrajectoryData, state = 1, s::Symbol) = 

    
 