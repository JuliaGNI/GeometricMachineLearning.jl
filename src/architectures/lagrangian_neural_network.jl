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

# `∇L`, `∇∇L` and `∇q̇∇q̇L` stood here: the gradient, the Hessian and the `q̇q̇` block of the Hessian
# of the network, each by a `Zygote.gradient` or `Zygote.hessian` over the input. They were described
# as the reference for what `LNNLoss` computes, but nothing called them and no test compared the two
# routes, so the reference was never taken. `LNNLoss` reaches the same quantities through
# `SymbolicNeuralNetworks.Jacobian`, because a nested `Zygote.gradient` inside a loss breaks the
# parameter gradient. Restoring the comparison means a test, not three uncalled definitions.
