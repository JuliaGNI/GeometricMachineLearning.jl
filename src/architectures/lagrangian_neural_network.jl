struct LagrangianNeuralNetwork{AT} <: Architecture
    dimin::Int
    width::Int
    nhidden::Int
    act::AT

    function LagrangianNeuralNetwork(dimin; width = dimin, nhidden = 1, activation = tanh)
        new{typeof(activation)}(dimin, width, nhidden, activation)
    end
end

@inline AbstractNeuralNetworks.dim(arch::LagrangianNeuralNetwork) = arch.dimin

function Chain(nn::LagrangianNeuralNetwork)
    inner_layers = Tuple(
        [Dense(nn.width, nn.width, nn.act) for _ in 1:nn.nhidden]
    )

    Chain(
        Dense(nn.dimin, nn.width, nn.act),
        inner_layers...,
        Linear(nn.width, 1; use_bias = false)
    )
end

# The three derivatives below take the `Zygote` route, which is the reference for what `LNNLoss`
# computes and not a second implementation of it: a nested `Zygote.gradient` inside a loss breaks
# the parameter gradient, so the loss reaches the same quantities through
# `SymbolicNeuralNetworks.Jacobian` instead. They are kept for checking that route by hand.

# gradient of the Lagrangian Neural Network
function ∇L(nn::NeuralNetwork{<:LagrangianNeuralNetwork}, x, params = params(nn))
    Zygote.gradient(x->sum(nn(x, params)), x)[1]
end

# hessian of the Lagrangian Neural Network
function ∇∇L(nn::NeuralNetwork{<:LagrangianNeuralNetwork}, q, q̇, params = params(nn))
    Zygote.hessian(x->sum(nn(x, params)), [q..., q̇...])
end

function ∇q̇∇q̇L(nn::NeuralNetwork{<:LagrangianNeuralNetwork}, q, q̇, params = params(nn))
    ∇∇L(nn, q, q̇, params)[(1 + length(q̇)):end, (1 + length(q̇)):end]
end
