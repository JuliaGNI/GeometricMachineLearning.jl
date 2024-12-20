
struct HamiltonianNeuralNetwork{AT} <: Architecture
    dimin::Int
    width::Int
    nhidden::Int
    act::AT

    function HamiltonianNeuralNetwork(dimin; width=dimin, nhidden=1, activation=tanh)
        new{typeof(activation)}(dimin, width, nhidden, activation)
    end
end

@inline AbstractNeuralNetworks.dim(arch::HamiltonianNeuralNetwork) = arch.dimin

function Chain(nn::HamiltonianNeuralNetwork)
    inner_layers = Tuple(
        [Dense(nn.width, nn.width, nn.act) for _ in 1:nn.nhidden]
    )

    Chain(
        Dense(nn.dimin, nn.width, nn.act),
        inner_layers...,
        Linear(nn.width, 1; use_bias = false)
    )
end

# gradient of the Hamiltonian Neural Network
gradient(nn::AbstractNeuralNetwork{<:HamiltonianNeuralNetwork}, x, params = params(nn)) = Zygote.gradient(ξ -> sum(nn(ξ, params)), x)[1]

# vector field of the Hamiltonian Neural Network
function vectorfield(nn::AbstractNeuralNetwork{<:HamiltonianNeuralNetwork}, x, params = params(nn)) 
    n_dim = length(x)÷2
    I = Diagonal(ones(n_dim))
    Z = zeros(n_dim,n_dim)
    symplectic_matrix = [Z I;-I Z]
    return symplectic_matrix * gradient(nn, x, params)
end

