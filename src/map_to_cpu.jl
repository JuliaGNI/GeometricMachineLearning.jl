# Move a set of parameters, or a whole network, from a device back to the host.
#
# One walk covers every leaf. `mapstorage` hands `f` the `freeparameters` of a leaf and `rebuild`s the
# leaf around the result, so a `StiefelManifold` comes back a `StiefelManifold` and a `SymmetricMatrix`
# keeps its `n`.
# `GeometricOptimizers` supplies the protocol for its own structured types, so nothing here has to know
# which of them exist, and a type added upstream is covered without a change on this side.
#
# `mapstorage` and not `mapparameters`: the latter hands `f` *whole* leaves, which would still need one
# method per structured type to reach the storage.
#
# A network lands on `CPU()`, and its fourth type parameter says so: a copy whose leaves are host
# arrays but whose backend is still the device it came from would send its next forward pass to that
# device with host data.
_to_host(A::AbstractArray{T}) where {T} = Array{T}(A)

map_to_cpu(ps) = mapstorage(_to_host, ps)

function map_to_cpu(nn::NeuralNetwork{AT, MT}) where {AT, MT}
    ps = map_to_cpu(params(nn))
    NeuralNetwork{AT, MT, typeof(ps), CPU}(nn.architecture, nn.model, ps, CPU())
end
