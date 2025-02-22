function map_to_cpu(ps::NeuralNetworkParameters) 
    NeuralNetworkParameters(NamedTuple{keys(ps)}(Tuple(map_to_cpu(ps[key]) for key in keys(ps))))
end

map_to_cpu(layer::NamedTuple) = apply_toNT(map_to_cpu, layer)

function map_to_cpu(A::AbstractArray{T}) where T
    Array{T}(A)
end

function map_to_cpu(Y::StiefelManifold{T}) where T 
    StiefelManifold(Array{T}(Y.A))
end

function map_to_cpu(U::UpperTriangular{T}) where T
    UpperTriangular(Array{T}(U.S), U.n)
end

function map_to_cpu(L::LowerTriangular{T}) where T
    LowerTriangular(Array{T}(L.S), L.n)
end

function map_to_cpu(A::SkewSymMatrix{T}) where T
    SkewSymMatrix(Array{T}(A.S), A.n)
end

function map_to_cpu(A::SymmetricMatrix{T}) where T 
    SymmetricMatrix(Array{T}(A.S), A.n)
end

function map_to_cpu(nn::NeuralNetwork{AT, MT, <:Any, BT}) where {AT, MT, BT}
    ps = map_to_cpu(params(nn))
    NeuralNetwork{AT, MT, typeof(ps), BT}(nn.architecture, nn.model, ps, nn.backend)
end