
struct HamiltonianNeuralNetwork{AT} <: Architecture
    dimin::Int
    width::Int
    nhidden::Int
    act::AT

    function HamiltonianNeuralNetwork(dimin; width=dimin, nhidden=1, activation=tanh)
        new{typeof(activation)}(dimin, width, nhidden, activation)
    end
end

@inline dim(arch::HamiltonianNeuralNetwork) = arch.dimin

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


# evaulation of the Hamiltonian Neural Network
(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork})(x, params = nn.params) = sum(apply(nn, x, params))

# gradient of the Hamiltonian Neural Network
gradient(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, x, params = nn.params) = Zygote.gradient(ξ -> nn(ξ, params), x)[1]

# vector field of the Hamiltonian Neural Network
function vectorfield(nn::LuxNeuralNetwork{<:HamiltonianNeuralNetwork}, x, params = nn.params) 
    n_dim = length(x)÷2
    I = Diagonal(ones(n_dim))
    Z = zeros(n_dim,n_dim)
    symplectic_matrix = [Z I;-I Z]
    return symplectic_matrix * gradient(nn, x, params)
end

