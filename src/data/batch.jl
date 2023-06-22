
# Get_batch functions

function get_batch(data::DataTrajectory, batch_size_t::Tuple{Int64,Int64,Int64})

    batch_nb_trajectory, batch_size, size_sequence = batch_size_t
    
    l = data.get_nb_trajectory()
    index_trajectory = rand(1:l, min(batch_nb_trajectory,l))

    index_qp = []
    for i in index_trajectory
        l_i = data.get_length_trajectory(i)
        push!(index_qp, [(i,j) for j in rand(1:l_i-size_sequence+1, min(l_i, batch_size)÷size_sequence)])
    end

    index_batch= vcat(index_qp...)

    return index_batch
end

get_batch(data::DataTrajectory) = vcat([[(i,j) for j in 1:2:data.get_length_trajectory(i)-1] for i in 1:1:data.get_nb_trajectory()]...)

get_batch(data::DataSampled, batch_size::Int = data.get_nb_point()) = rand(1:data.get_nb_point(), batch_size)

get_batch(data::DataTarget, batch_size_t::Union{Tuple{Int64,Int64,Int64},Int64}) = get_batch(data.data, batch_size_t)

get_batch(data::DataTarget) = get_batch(data.data)


const DEFAULT_BATCH_SIZE = 10
const DEFAULT_BATCH_NB_TAJECTORY= 1

default_index_batch(::DataTrajectory) = (1, DEFAULT_BATCH_SIZE, 2)
default_index_batch(::DataSampled) = DEFAULT_BATCH_SIZE
default_index_batch(datat::DataTarget) = default_index_batch(datat.data)

