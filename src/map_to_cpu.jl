map_to_cpu(ps::Tuple) = Tuple([map_to_cpu(layer) for layer in ps])
map_to_cpu(layer::NamedTuple) = apply_toNT(map_to_cpu, layer)
function map_to_cpu(A::AbstractArray{T}) where T
    Array{T}(A)
end

function map_to_cpu(Y::StiefelManifold{T}) where T 
    StiefelManifold(Array{T}(Y.A))
end

function map_to_cpu(nn::NeuralNetwork{AT, MT}) where {AT, MT}
    ps = map_to_cpu(nn.params)
    NeuralNetwork{AT, MT, typeof(ps)}(nn.architecture, nn.model, ps)
end