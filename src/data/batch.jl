const DEFAULT_BATCH_SIZE = 10
const DEFAULT_BATCH_NB_TAJECTORY= 1

min_length_batch(ti::AbstractTrainingIntegrator) = 1


default_index_batch(::AbstractTrainingIntegrator{<:AbstractDataSymbol, <:SampledData})  = DEFAULT_BATCH_SIZE
default_index_batch(ti::AbstractTrainingIntegrator{<:AbstractDataSymbol, <:TrajectorydData})  = (1, DEFAULT_BATCH_SIZE, min_length_batch(ti))
default_index_batch(::TrainingData{T,TrajectoryData} where T, ti::AbstractTrainingIntegrator) = (1, DEFAULT_BATCH_SIZE, min_length_batch(ti))
default_index_batch(::TrainingData{T,SampledData} where T, ::AbstractTrainingIntegrator) = DEFAULT_BATCH_SIZE


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

function get_batch(data::TrainingData{T,TrajectoryData} where T, batch_size_t::Tuple{Int64,Int64,Int64})

    batch_nb_trajectory, batch_size, size_sequence = batch_size_t
    
    l = get_nb_trajectory(data)
    index_trajectory = rand(1:l, min(batch_nb_trajectory,l))

    index_qp = []
    for i in index_trajectory
        l_i = get_length_trajectory(data, i)
        push!(index_qp, [(i,j) for j in rand(1:l_i-size_sequence+1, min(l_i, batch_size)÷size_sequence)])
    end

    index_batch= vcat(index_qp...)

    return index_batch
end

get_batch(data::TrainingData{T,TrajectoryData} where T) = vcat([[(i,j) for j in 1:2:get_length_trajectory(data, i)-1] for i in 1:1:get_nb_trajectory(data)]...)

#=
    The get_batch function for data with shape of  SampledData gives just a subsequence of 1:get_nb_point(data) of selected points to form
    the batch index.
=#

function get_batch(data::TrainingData{T,SampledData} where T, batch_size::Int = get_nb_point(data)) 
    
    @assert get_nb_point(data) >= batch_size

    rand(1:get_nb_point(data), batch_size)
end

get_batch(data::TrainingData{T,SampledData} where T) = 1:get_nb_point(data)






