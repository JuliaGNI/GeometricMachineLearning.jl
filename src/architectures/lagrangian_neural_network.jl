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

# The gradient, the Hessian and the `q̇q̇` block of the Hessian of this network are `LNNLoss`'s
# business, and it reaches them through `SymbolicNeuralNetworks.Jacobian` rather than through
# `Zygote`: a nested `Zygote.gradient` inside a loss breaks the parameter gradient.
