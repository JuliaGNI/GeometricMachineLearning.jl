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
    get_q::Function
    get_p::Function
    
    function data_trajectory(Data, Get_nb_trajectory::Function, Get_length_trajectory::Function, Get_q::Function, Get_p::Function, Get_Δt::Function = NothingFunction())
        get_Δt() = Get_Δt(Data)
        get_nb_trajectory() = Get_nb_trajectory(Data)
        get_length_trajectory(i) = Get_length_trajectory(Data, i)
        get_q(i,n) = Get_q(Data,i,n)
        get_p(i,n) = Get_p(Data,i,n)
        new(get_Δt, get_nb_trajectory, get_length_trajectory, get_q, get_p)
    end
end


struct data_sampled <: Training_data
    get_nb_point::Function
    get_q::Function
    get_p::Function
    
    function data_sampled(Data, Get_nb_point::Function, Get_q::Function, Get_p::Function)
        get_nb_point() = Get_nb_point(Data)
        get_q(n) = Get_q(Data,n)
        get_p(n) = Get_p(Data,n)
        new(get_nb_point, get_q, get_p)
    end
end


struct dataTarget{T<:Training_data} <: Training_data
    data::T
    get_q̇::Function
    get_ṗ::Function

    function dataTarget(data::data_trajectory, Target, Get_q̇::Function, Get_ṗ::Function)
        get_q̇(i,n) = Get_q̇(Target,i,n)
        get_ṗ(i,n) = Get_ṗ(Target,i,n)
        new{typeof(data)}(data, get_q̇, get_ṗ)
    end

    function dataTarget(data::data_sampled, Target, Get_q̇::Function, Get_ṗ::Function)
        get_q̇(n) = Get_q̇(Target,n)
        get_ṗ(n) = Get_ṗ(Target,n)
        new{typeof(data)}(data, get_q̇, get_ṗ)
    end
end


# Get_batch functions

function get_batch(data::data_trajectory, batch_size_t::Tuple{Int64,Int64})

    batch_nb_trajectory, batch_size = batch_size_t
    
    l = data.get_nb_trajectory()
    index_trajectory = rand(1:l, min(batch_nb_trajectory,l))

    index_qp = []
    for i in index_trajectory
        l_i = data.get_length_trajectory(i)
        push!(index_qp, [(i,j) for j in rand(1:l_i-1, min(l_i, batch_size)÷2)])
    end

    index_batch= vcat(index_qp...)

    return index_batch
end

get_batch(data::data_trajectory) = vcat([[(i,j) for j in 1:2:data.get_length_trajectory(i)-1] for i in 1:1:data.get_nb_trajectory()]...)

get_batch(data::data_sampled, batch_size::Int = data.get_nb_point()) = rand(1:data.get_nb_point(), batch_size)

get_batch(data::dataTarget, batch_size_t::Union{Tuple{Int64,Int64},Int64}) = get_batch(data.data, batch_size_t)

get_batch(data::dataTarget) = get_batch(data.data)



const DEFAULT_BATCH_SIZE = 10
const DEFAULT_BATCH_NB_TAJECTORY= 1
default_index_batch(::data_trajectory) = (1, DEFAULT_BATCH_SIZE)
default_index_batch(::data_sampled) = DEFAULT_BATCH_SIZE
default_index_batch(datat::dataTarget) = default_index_batch(datat.data)
