struct NothingFunction <: Function end
(::NothingFunction)(args...) = nothing
is_NothingFunction(f::Function) = typeof(f)==NothingFunction


function Base.:+(a::Float64, b::Tuple{Float64})
    x, = b
    return a+x
end

function Base.:+(a::Vector{Float64}, b::Tuple{Float64})
    x, = b
    y, = a
    return y+x
end

Zygote.OneElement(t1::Tuple{Float64}, t2::Tuple{Int64}, t3::Tuple{Base.OneTo{Int64}}) = Zygote.OneElement(t1[1], t2, t3)

#Data structure

abstract type Training_data end


# Data structure 

struct data_trajectory <: Training_data
    get_Δt::Function
    get_nb_trajectory::Function
    get_length_trajectory::Function
    get_data::Dict{Symbol, <:Function}
    
    function data_trajectory(Data, Get_nb_trajectory::Function, Get_length_trajectory::Function,  Get_data::Dict{Symbol, <:Function}, Get_Δt::Function = NothingFunction())
        get_Δt() = Get_Δt(Data)
        get_nb_trajectory() = Get_nb_trajectory(Data)
        get_length_trajectory(i) = Get_length_trajectory(Data, i)
        get_data = Dict([(key, (i,n)->value(Data,i,n)) for (key,value) in Get_data])
        new(get_Δt, get_nb_trajectory, get_length_trajectory, get_data)
    end

    function data_trajectory(Data, Get_data::Dict{Symbol, <:Function})
        
        @assert haskey(Get_data, :Δt)
        @assert haskey(Get_data, :nb_trajectory)
        @assert haskey(Get_data, :length_trajectory)

        get_Δt() = Get_data[:Δt](Data)
        get_nb_trajectory() = Get_data[:nb_trajectory](Data)
        get_length_trajectory(i) = Get_data[:length_trajectory](Data, i)
       
        get_data = Dict([(key, (i,n)->value(Data,i,n)) for (key,value) in Get_data])

        delete!(get_data, :Δt)
        delete!(get_data, :nb_trajectory)
        delete!(get_data, :length_trajectory)

        new(get_Δt, get_nb_trajectory, get_length_trajectory, get_data)
    end

end

struct data_sampled <: Training_data
    get_nb_point::Function
    get_data::Dict{Symbol, <:Function}

    function data_sampled(Data, Get_nb_point::Function, Get_data::Dict{Symbol, <:Function})
        get_nb_point() = Get_nb_point(Data)
        get_data = Dict([(key, (i,n)->value(Data,i,n)) for (key,value) in Get_data])
        new(get_nb_point, get_data)
    end
    
    function data_sampled(Data, Get_data::Dict{Symbol, <:Function})
            
        @assert haskey(Get_data, :nb_points)
        get_nb_point() = Get_data[:nb_points](Data)

        get_data = Dict([(key, n->value(Data,n)) for (key,value) in Get_data])
        
        delete!(get_data, :nb_points)

        new(get_nb_point, get_data)
    end

end


struct dataTarget{T<:Training_data} <: Training_data
    data::T
    get_data::Dict{Symbol, <:Function}
    get_target::Dict{Symbol, <:Function}

    function dataTarget(data::data_trajectory, Target, Get_target::Dict{Symbol, <:Function})
        get_target = Dict([(key, (i,n)->value(Target,i,n)) for (key,value) in Get_target])
        new{typeof(data)}(data, data.get_data, get_target)
    end
    
    function dataTarget(data::data_sampled, Target, Get_target::Dict{Symbol, <:Function})
        get_target = Dict([(key, n->value(Target,n)) for (key,value) in Get_target])
        new{typeof(data)}(data, data.get_data, get_target)
    end
end


# Get_batch functions

function get_batch(data::data_trajectory, batch_size_t::Tuple{Int64,Int64,Int64})

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

get_batch(data::data_trajectory) = vcat([[(i,j) for j in 1:2:data.get_length_trajectory(i)-1] for i in 1:1:data.get_nb_trajectory()]...)

get_batch(data::data_sampled, batch_size::Int = data.get_nb_point()) = rand(1:data.get_nb_point(), batch_size)

get_batch(data::dataTarget, batch_size_t::Union{Tuple{Int64,Int64,Int64},Int64}) = get_batch(data.data, batch_size_t)

get_batch(data::dataTarget) = get_batch(data.data)



const DEFAULT_BATCH_SIZE = 10
const DEFAULT_BATCH_NB_TAJECTORY= 1
default_index_batch(::data_trajectory) = (1, DEFAULT_BATCH_SIZE, 3)
default_index_batch(::data_sampled) = DEFAULT_BATCH_SIZE
default_index_batch(datat::dataTarget) = default_index_batch(datat.data)
