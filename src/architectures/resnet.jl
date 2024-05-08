struct ResNet{AT} <: NeuralNetworkIntegrator
    sys_dim::Int 
    n_blocks::Int 
    activation::AT
end

function Chain(arch::ResNet{AT}) where AT
    layers = ()
    for _ in 1:arch.n_blocks 
        # nonlinear layers
        layers = (layers..., ResNetLayer(arch.sys_dim, arch.activation; use_bias=true))
    end

    # linear layers for the output
    layers = (layers..., ResNetLayer(arch.sys_dim, identity; use_bias=true))

    Chain(layers...)
end