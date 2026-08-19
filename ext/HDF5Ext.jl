module HDF5Ext

using HDF5
using GeometricMachineLearning
import AbstractNeuralNetworks: changebackend, NeuralNetworkBackend, Architecture
# `save`, `load`, `params` and the parameter container are `NeuralNetworkParameters`' as of
# `AbstractNeuralNetworks` 0.7, which only re-binds them; reach for them where they are defined.
import NeuralNetworkParameters: NetworkParameters, params, save, load

# A `NeuralNetworkParameters` nested inside a parameter tree -- the parameter-dependent
# architectures put one per sub-network. AbstractNeuralNetworks has `save(::H5DataStore,
# ::NeuralNetworkParameters)` for the top level only.
#
# TODO: type piracy -- `h5save` and `NeuralNetworkParameters` are both AbstractNeuralNetworks'.
# This belongs in ANN's own `ext/HDF5Ext.jl`, next to `h5save(::H5DataStore, ::NamedTuple, …)`.
function h5save(h5::HDF5.Group, p::NeuralNetworkParameters, path::AbstractString)
    h5save(h5, params(p), path)
end

# ---------------------------------------------------------------------------
# changebackend — new methods for GML special array types
#
# AbstractNeuralNetworks.changebackend handles AbstractArray and NamedTuple.
# Moving a NeuralNetwork between devices fails for parameters that include
# StiefelManifold, SymmetricMatrix, or SkewSymMatrix without these methods.
#
# `changebackend` is `AbstractNeuralNetworks`' and the types are `GeometricOptimizers`', so these
# methods are piracy the same way the `h5save` ones were before `GeometricOptimizers` took over the
# leaf protocol. They belong in a `GeometricOptimizers` extension on `AbstractNeuralNetworks`; that
# is a separate change with its own release chain, so they stay here for now.
# ---------------------------------------------------------------------------

function changebackend(backend::NeuralNetworkBackend, Y::StiefelManifold)
    StiefelManifold(changebackend(backend, Y.A))
end

function changebackend(backend::NeuralNetworkBackend, A::SymmetricMatrix)
    SymmetricMatrix(changebackend(backend, A.S), A.n)
end

function changebackend(backend::NeuralNetworkBackend, A::SkewSymMatrix)
    SkewSymMatrix(changebackend(backend, A.S), A.n)
end

function changebackend(backend::NeuralNetworkBackend, A::LowerTriangular)
    LowerTriangular(changebackend(backend, A.S), A.n)
end

function changebackend(backend::NeuralNetworkBackend, A::UpperTriangular)
    UpperTriangular(changebackend(backend, A.S), A.n)
end

# ---------------------------------------------------------------------------
# save / load — the entry points that dispatch on this package's `NeuralNetwork`.
#
# The traversal itself is not here. `NeuralNetworkParameters` walks the parameter set and records
# each group's key order; `GeometricOptimizers` says through `freeparameters`/`rebuild` where each
# structured matrix keeps its numbers, and registers the types so a file loads with no prototype.
# ---------------------------------------------------------------------------

"""
    save(h5::HDF5.H5DataStore, nn::NeuralNetwork)

Save the parameters of `nn` into an already-open HDF5 store.

Extends `save` with a dispatch on `NeuralNetwork`. The parameters themselves are written by
`NeuralNetworkParameters`, which tags each structured leaf with the type to rebuild it as and
records the key order of every group.
"""
save(h5::HDF5.H5DataStore, nn::NeuralNetwork) = save(h5, params(nn))

"""
    save(filename::AbstractString, nn::NeuralNetwork)

Convenience overload: open `filename` for writing, call [`save`](@ref) on the store, and return
`filename`.
"""
function save(filename::AbstractString, nn::NeuralNetwork)
    HDF5.h5open(filename, "w") do h5
        save(h5, nn)
    end
    filename
end

"""
    load(::Type{NeuralNetwork}, h5::HDF5.H5DataStore, arch::Architecture; backend = CPU())
    load(::Type{NeuralNetwork}, h5::HDF5.H5DataStore, arch::Architecture, prototype; backend = CPU())

Load network parameters from an already-open HDF5 store and return a `NeuralNetwork` for `arch`.

The element type is whatever the file holds, so a `Float32` network reloads as `Float32`.

Structured parameters — `StiefelManifold`, `SymmetricMatrix` and the rest — are rebuilt from the
type each was stored under, which `GeometricOptimizers` registers with
`NeuralNetworkParameters.register_parameter_type!`. Pass `prototype`, a parameter set of the right
shape, to rebuild against it instead and skip the registry altogether.
"""
function load(::Type{NeuralNetwork}, h5::HDF5.H5DataStore, arch::Architecture;
              backend::NeuralNetworkBackend = CPU())
    NeuralNetwork(arch, Chain(arch), load(NetworkParameters, h5), backend)
end

function load(::Type{NeuralNetwork}, h5::HDF5.H5DataStore, arch::Architecture, prototype;
              backend::NeuralNetworkBackend = CPU())
    NeuralNetwork(arch, Chain(arch), load(NetworkParameters, h5, prototype), backend)
end

"""
    load(::Type{NeuralNetwork}, filename::AbstractString, arch::Architecture; backend = CPU())
    load(::Type{NeuralNetwork}, filename::AbstractString, arch::Architecture, prototype; backend = CPU())

Convenience overload: open `filename` for reading, then call
[`load`](@ref) on the store.
"""
function load(::Type{NeuralNetwork}, filename::AbstractString, arch::Architecture;
              backend::NeuralNetworkBackend = CPU())
    HDF5.h5open(filename, "r") do h5
        load(NeuralNetwork, h5, arch; backend = backend)
    end
end

function load(::Type{NeuralNetwork}, filename::AbstractString, arch::Architecture, prototype;
              backend::NeuralNetworkBackend = CPU())
    HDF5.h5open(filename, "r") do h5
        load(NeuralNetwork, h5, arch, prototype; backend = backend)
    end
end

end
