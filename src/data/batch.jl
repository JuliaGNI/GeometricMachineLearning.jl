const DEFAULT_BATCH_SIZE = 10
const DEFAULT_BATCH_NB_TAJECTORY= 2

min_length_batch(ti::AbstractTrainingIntegrator) = 1

complete_batch_size(::TrainingData{T,TrajectoryData} where T, ti::AbstractTrainingIntegrator, bs::Tuple{Int64, Int64, Int64}) = bs
complete_batch_size(::TrainingData{T,TrajectoryData} where T, ti::AbstractTrainingIntegrator, bs::Tuple{Int64, Int64}) = (bs..., min_length_batch(ti))
complete_batch_size(data::TrainingData{T,TrajectoryData} where T, ti::AbstractTrainingIntegrator, bs::Tuple{Int64}) = (get_nb_trajectory(data), bs..., min_length_batch(ti))
complete_batch_size(data::TrainingData{T,TrajectoryData} where T, ti::AbstractTrainingIntegrator, bs::Int64) = (get_nb_trajectory(data), bs, min_length_batch(ti))
complete_batch_size(data::TrainingData{T,TrajectoryData} where T, ti::AbstractTrainingIntegrator, ::Missing) = (1, min(DEFAULT_BATCH_SIZE, min_length(data) ), min_length_batch(ti))
complete_batch_size(::TrainingData{T,SampledData} where T, ::AbstractTrainingIntegrator, bs::Int64) = bs
complete_batch_size(data::TrainingData{T,SampledData} where T, ::AbstractTrainingIntegrator, ::Missing) = min(DEFAULT_BATCH_SIZE, get_nb_point(data))


#=
    get_batch function
=#


function get_batch(::AbstractTrainingData, ::Any) 
    throw(ArgumentError("No match between the shape of data and the shape of batch_size."))
end


#=
    The get_batch function for data with shape of  TrajectoryData is designed to give a vector of tuple of index to select a subset of the data.
    Arguments are :
        - data : the TrainingData with shape TrajectoryData,
        - batch_size_t : a 3-Tuple of integer (n,m,p) where
                                                    - n is the number of trajectories present in the batch index,
                                                    - m is number of points choosen in each selected trajectory,
                                                    - p is the minimal size of the sequence of points needed for an integration 
                                                        (ex if qₙ and qₙ₊₁ are needed then p=2).
    The trajectories selected and the points are randomly choosen with uniform law.  
=#

function get_batch(data::TrainingData{T,TrajectoryData} where T, batch_size_t::Tuple{Int64,Int64,Int64}; check = true)

    batch_nb_trajectory, batch_size, size_sequence = batch_size_t

    check ? check_batch_size(data, batch_size_t) : nothing

    l = get_nb_trajectory(data)
    index_trajectory = rand(1:l, min(batch_nb_trajectory, l))

    index_qp = []
    for i in index_trajectory
        l_i = get_length_trajectory(data, i)
        push!(index_qp, [(i,j) for j in rand(1:l_i-size_sequence+1, min(l_i, batch_size)÷size_sequence)])
    end

    index_batch= vcat(index_qp...)

    return index_batch
end

get_batch(data::TrainingData{T,TrajectoryData} where T, batch_size::Tuple{Int64, Int64}; kwargs...) = get_batch(data, (batch_size...,1); kwargs...)

get_batch(data::TrainingData{T,TrajectoryData} where T, batch_size::Int64; kwargs...) = get_batch(data, (get_nb_trajectory(data),batch_size,1); kwargs...)
    

#=
    The get_batch function for data with shape of  SampledData gives just a subsequence of 1:get_nb_point(data) of selected points to form
    the batch index.
=#

function get_batch(data::TrainingData{T,SampledData} where T, batch_size::Int; check = true) 
    
    check ? check_batch_size(data, batch_size) : nothing

    rand(1:get_nb_point(data), min(batch_size, get_nb_point(data)))
end


#=
    The purpose of the check_batch_size function is to check that the batch can be processed according to the data (for example, whether there is enough data).
    and to warn the user of any automatic correction if necessary.
=#


function check_batch_size(data::TrainingData{T,TrajectoryData} where T, batch_size::Tuple{Int64,Int64,Int64})
    batch_nb_trajectory, batch_size, size_sequence = batch_size
    nb_trajectory = get_nb_trajectory(data)
    if batch_nb_trajectory > nb_trajectory
        @warn "The number of trajectory in the batch is greater than the number of trajectory in data.\
            \nTherefore, the batch is done on "*string(nb_trajectory)*" trajectories instead of "*string(batch_nb_trajectory)*"."
    end
    nb = length([i for i in 1: nb_trajectory if get_length_trajectory(data, i) <  batch_size])
    if nb > 0
        @warn "The size of data to take inside each selected trajectory is greater than the size of "*string(nb)*"/"*string(nb_trajectory)*" trajectories in data.\
            \nFor those, "*string(batch_size)*" is replaced by their respectives sizes."
    end
    nb = length([i for i in 1: nb_trajectory if get_length_trajectory(data, i) < size_sequence])
    if  nb ==  nb_trajectory
        0
        #throw(ArgumentError("The minimal size required for sub-trajectories is greater than any trajectories of the data.\ 
        #    Therefore, it is not possible to proceed with the batch."))
    end
    if nb>0
        @warn "The minimal size required for sub-trajectories is greater than the size of "*string(nb)*"/"*string(nb_trajectory)*" trajectories in data.\
        \nTherefore, the batch is done on theses trajectories instead."
    end
end

function check_batch_size(data::TrainingData{T,SampledData} where T, batch_size::Int64)
     if batch_size > get_nb_point(data)
        @warn "The size of batch is greater than the size of data.\
        \nTherefore, the batch is done on "*string(get_nb_point(data))*" trajectories instead of "*string(batch_size)*"."
        return get_nb_point(data)
     end
end



