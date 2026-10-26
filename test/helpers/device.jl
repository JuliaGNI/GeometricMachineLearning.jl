# A parameter set of the host, moved onto a device array.
#
# `changebackend(JLBackend(), ps)` cannot do this: it copies through
# `KernelAbstractions.copyto!(backend, dest, src)`, and `AbstractNeuralNetworks`' method for that call
# has no `JLArrays` method, so the copy raises a `MethodError` whose message names `copyto!` and not
# `changebackend`. The entry that records the defect is in `KNOWN_ISSUES.md`.
#
# The walk is over the two levels of the set, so every structured type needs the constructor that
# rebuilds it around device storage: a `StiefelManifold` around its `.A`, and a `SymmetricMatrix`, a
# `SkewSymMatrix`, a `StrictlyLowerTriangular` and a `StrictlyUpperTriangular` around `.S` and their
# `n`. A leaf may itself be a `NamedTuple`, as a transformer's attention heads are, and `map` reaches
# the arrays in it.

using GeometricMachineLearning
using GeometricMachineLearning: NetworkParameters, params, SymmetricMatrix, SkewSymMatrix,
                                StrictlyLowerTriangular, StrictlyUpperTriangular,
                                StiefelManifold
using JLArrays: JLArray, JLBackend
import KernelAbstractions

to_device(x::AbstractArray) = JLArray(x)
to_device(x::StiefelManifold) = StiefelManifold(JLArray(x.A))
to_device(x::SymmetricMatrix) = SymmetricMatrix(JLArray(x.S), x.n)
to_device(x::SkewSymMatrix) = SkewSymMatrix(JLArray(x.S), x.n)
to_device(x::StrictlyLowerTriangular) = StrictlyLowerTriangular(JLArray(x.S), x.n)
to_device(x::StrictlyUpperTriangular) = StrictlyUpperTriangular(JLArray(x.S), x.n)
to_device(x::NamedTuple) = map(to_device, x)

# The two levels of a parameter set: each layer's leaves, and the set rebuilt around them.
function to_device(ps::NetworkParameters)
    names = keys(ps)
    NetworkParameters(NamedTuple{names}(map(name -> map(to_device, getproperty(ps, name)), names)))
end

# The network `arch` of the host, with its parameter set on the device. A `StiefelLayer` is built
# inside a `Chain`, because a bare model's `initialparameters` is a `NamedTuple` and
# `NeuralNetwork(model, …)` refuses one.
function network_on_device(arch, ::Type{T}) where {T}
    nn = NeuralNetwork(arch, CPU(), T)
    NeuralNetwork(nn.architecture, nn.model, to_device(params(nn)), JLBackend())
end

# The backend a leaf lives on. `parent` of a plain array is the array itself, and `parent` of a
# `Manifold` or of a structured matrix is its storage.
leaf_backend(x) = KernelAbstractions.get_backend(parent(x))

# Every leaf of `g` is an array of the device and of element type `T`. A `NamedTuple` leaf is walked
# into: its arrays sit at the third level, which `foreach_leaf` does not reach.
function leaves_on_device(g, ::Type{T}) where {T}
    ok = Ref(true)
    walk(x::NamedTuple) = foreach(k -> walk(x[k]), keys(x))
    walk(x::AbstractArray) = (ok[] &= parent(x) isa JLArray && eltype(parent(x)) === T &&
                                      leaf_backend(x) isa JLBackend)
    for name in keys(g), k in keys(g[name])

        walk(g[name][k])
    end
    ok[]
end
