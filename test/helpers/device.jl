# A parameter set of the host, moved onto a device array.
#
# `changebackend(JLBackend(), ps)` cannot do this: it copies through
# `KernelAbstractions.copyto!(backend, dest, src)`, and `AbstractNeuralNetworks`' method for that call
# has no `JLArrays` method, so the copy raises a `MethodError` whose message names `copyto!` and not
# `changebackend`. The entry that records the defect is K31 in `KNOWN_ISSUES.md`.
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
using NeuralNetworkParameters: mapparameters
import KernelAbstractions
import Random

to_device(x::AbstractArray) = JLArray(x)
to_device(x::StiefelManifold) = StiefelManifold(JLArray(x.A))
to_device(x::SymmetricMatrix) = SymmetricMatrix(JLArray(x.S), x.n)
to_device(x::SkewSymMatrix) = SkewSymMatrix(JLArray(x.S), x.n)
to_device(x::StrictlyLowerTriangular) = StrictlyLowerTriangular(JLArray(x.S), x.n)
to_device(x::StrictlyUpperTriangular) = StrictlyUpperTriangular(JLArray(x.S), x.n)
to_device(x::NamedTuple) = map(to_device, x)

# The two levels of a parameter set: `mapparameters` walks each layer's leaves and rebuilds the set
# around them.
to_device(ps::NetworkParameters) = mapparameters(to_device, ps)

# The network `arch` of the host, with its parameter set on the device. A `StiefelLayer` is built
# inside a `Chain`, because a bare model's `initialparameters` is a `NamedTuple` and
# `NeuralNetwork(model, …)` refuses one.
function network_on_device(arch, ::Type{T}) where {T}
    nn = NeuralNetwork(arch, CPU(), T)
    NeuralNetwork(nn.architecture, nn.model, to_device(params(nn)), JLBackend())
end

# Two epochs of `Adam` on data of the device, the network afterwards, and its parameter set as it was
# before the first step: `train!` writes the new parameters in place, so the copy is what a step can be
# seen against. The batch is a keyword because each family draws its minibatches its own way; the loss
# is one because `train!` has no default one for a plain `Chain`; and the retraction is one because a
# manifold leaf of a `JLArray` needs `Geodesic()`.
function two_epochs(arch, ::Type{T}, data, n_epochs; batch, retraction = Cayley(),
        loss = nothing, autoencoder = false) where {T}
    Random.seed!(13)
    nn = network_on_device(arch, T)
    p₀ = deepcopy(params(nn))
    dl = DataLoader(JLArray(data); suppress_info = true, autoencoder)
    opt = TrainingOptimizer(params(nn); algorithm = Adam(), retraction)
    loss_kw = loss === nothing ? (;) : (; loss)
    history = train!(nn, opt, dl, batch, n_epochs; show_progress = false, loss_kw...)
    history, nn, p₀
end

# Whether a step moved the parameter set at all: a step that writes back what it was handed leaves
# every leaf as it was, and the assertions of a training testset would not see it. The difference is a
# reduction on the device, so no leaf is read one element at a time, and a leaf a network does not
# train (`nothing`) is skipped.
function parameters_moved(before, after)
    moved = Ref(false)
    differ(a, b) = !iszero(sum(abs, parent(b) .- parent(a)))
    changed(a::AbstractArray, b::AbstractArray) = (moved[] |= differ(a, b))
    changed(a, b) = nothing
    walk(a::NamedTuple, b::NamedTuple) = foreach(k -> walk(a[k], b[k]), keys(a))
    walk(a, b) = changed(a, b)
    for name in keys(after), k in keys(after[name])

        walk(before[name][k], after[name][k])
    end
    moved[]
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
