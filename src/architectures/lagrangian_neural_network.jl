
const DEFAULT_LNN_NRUNS = 1000


struct LagrangianNeuralNetwork{AT} <: Architecture
    dimin::Int
    width::Int
    nhidden::Int
    act::AT

    function LagrangianNeuralNetwork(dimin; width=dimin, nhidden=1, activation=tanh)
        new{typeof(activation)}(dimin, width, nhidden, activation)
    end
end

@inline dim(arch::LagrangianNeuralNetwork) = arch.dimin

function chain(nn::LagrangianNeuralNetwork, ::LuxBackend)
    inner_layers = Tuple(
        [Lux.Dense(nn.width, nn.width, nn.act) for _ in 1:nn.nhidden]
    )

    Lux.Chain(
        Lux.Dense(nn.dimin, nn.width, nn.act),
        inner_layers...,
        Lux.Dense(nn.width, 1; use_bias=false)
    )
end


# evaluation of the Lagrangian Neural Network
(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork})(x, params = nn.params) = sum(apply(nn, x, params))

# gradient of the Lagrangian Neural Network
∇L(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, x, params = nn.params) = Zygote.gradient(x->nn(x, params), x)[1]

# hessian of the Lagrangian Neural Network
∇∇L(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, q, q̇, params = nn.params) = Zygote.hessian(x->nn(x, params),[q...,q̇...])

∇q̇∇q̇L(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, q, q̇, params = nn.params) = ∇∇L(nn, q, q̇, params)[(1+length(q̇)):end,(1+length(q̇)):end] 

∇q∇q̇L(nn::LuxNeuralNetwork{<:LagrangianNeuralNetwork}, q, q̇, params = nn.params) = ∇∇L(nn, q, q̇, params)[1:length(q),(1+length(q̇)):end] 





