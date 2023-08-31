# Structure
abstract type SympNet{AT} <: Architecture end

struct LASympNet{AT,T1,T2,T3} <: SympNet{AT} 
    dim::Int
    width::Int
    nhidden::Int
    act::AT
    init_uplow_linear::Vector{Bool}
    init_uplow_act::Vector{Bool}
    init_sym_matrices::T1
    init_bias::T2
    init_weight::T3

    function LASympNet(dim; width=9, nhidden=0, activation=tanh, init_uplow_linear=[true,false], init_uplow_act=[true,false],init_sym_matrices=Lux.glorot_uniform, init_bias=Lux.zeros32, init_weight=Lux.glorot_uniform) 
        new{typeof(activation),typeof(init_sym_matrices),typeof(init_bias),typeof(init_weight)}(dim, min(width,9), nhidden, activation, init_uplow_linear, init_uplow_act, init_sym_matrices, init_bias, init_weight)
    end

end

@inline AbstractNeuralNetworks.dim(arch::LASympNet) = arch.dim

struct GSympNet{AT} <: SympNet{AT} 
    dim::Int
    width::Int
    nhidden::Int
    act::AT
    init_uplow::Vector{Bool}

    function GSympNet(dim; width=dim, nhidden=2, activation=tanh, init_uplow=[true,false]) 
        new{typeof(activation)}(dim, width, nhidden, activation, init_uplow,)
    end
end


@inline dim(arch::GSympNet) = arch.dim

# Chain function
function Chain(nn::GSympNet)
    inner_layers = Tuple(
        [Gradient(nn.dim, nn.width, nn.act, change_q = nn.init_uplow[Int64((i-1)%length(nn.init_uplow)+1)]) for i in 1:nn.nhidden]
    )
    Chain(
        inner_layers...
    )
end

function Chain(nn::LASympNet)
    couple_layers = []
    for i in 1:nn.nhidden
        for j in 1:nn.width
            push!(couple_layers, LinearSymplectic(nn.dim, change_q = nn.init_uplow_linear[Int64((i-1+j-1)%length(nn.init_uplow_linear)+1)], bias=(j==nn.width), init_weight=nn.init_sym_matrices, init_bias=nn.init_bias))
        end
        push!(couple_layers,Gradient(nn.dim, nn.dim, nn.act, full_grad = false, change_q = nn.init_uplow_act[Int64((i-1)%length(nn.init_uplow_act)+1)]))
    end

    for j in 1:nn.width
        push!(couple_layers, LinearSymplecticLayer(nn.dim, change_q = nn.init_uplow_linear[Int64((nn.nhidden+1-1+j-1)%length(nn.init_uplow_linear)+1)], bias=(j==nn.width), init_weight=nn.init_sym_matrices, init_bias=nn.init_bias))
    end
    
    Chain(
        couple_layers...
    )
end


# Results

function Iterate_Sympnet(nn::NeuralNetwork{<:SympNet}, q0, p0; n_points = DEFAULT_SIZE_RESULTS)

    n_dim = length(q0)
    
    # Array to store the predictions
    q_learned = zeros(n_points,n_dim)
    p_learned = zeros(n_points,n_dim)
    
    # Initialisation
    q_learned[1,:] = q0
    p_learned[1,:] = p0
    
    #Computation of phase space
    for i in 2:n_points
        qp_learned =  nn([q_learned[i-1,:]..., p_learned[i-1,:]...])
        q_learned[i,:] = qp_learned[1:n_dim]
        p_learned[i,:] = qp_learned[(1+n_dim):end]
    end

    return q_learned, p_learned
end