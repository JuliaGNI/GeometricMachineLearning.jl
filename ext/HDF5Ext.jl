module HDF5Ext

using HDF5
using GeometricMachineLearning
import AbstractNeuralNetworks: changebackend, NeuralNetworkBackend, save, load, Architecture
import NeuralNetworkParameters: NetworkParameters, params

# ---------------------------------------------------------------------------
# The traversal is not here any more.
#
# This extension used to carry five `h5save` methods that tagged a `gml_type` attribute, plus
# `_gml_h5load` and `_natural_sort_keys` to read them back. All three jobs now belong to packages
# that own the pieces:
#
#   * `NeuralNetworkParameters` walks the parameter set and writes it, recording each group's key
#     order in a `keys` attribute — which is what the `_natural_sort_keys` heuristic here was
#     standing in for, and it guessed rather than knowing. Names that do not end in a digit were
#     sorted lexicographically and silently came back in the wrong order.
#
#   * `GeometricOptimizers` says where each structured matrix keeps its numbers, through
#     `freeparameters`/`rebuild`, and registers the types so a file loads with no prototype.
#     `StiefelManifold` and `SymmetricMatrix` are its types, not this package's, so the methods
#     were type piracy here — on `h5save` and on the type both.
#
# Files written by the old code still load: `NeuralNetworkParameters` recognises the `gml_type`
# tag and rebuilds through the same registry (see `test/hdf5_support.jl`).
#
# What is left is the two entry points that genuinely dispatch on this package's `NeuralNetwork`.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# changebackend — new methods for GML special array types
#
# AbstractNeuralNetworks.changebackend handles AbstractArray and NamedTuple.
# Moving a NeuralNetwork between devices fails for parameters that include
# StiefelManifold, SymmetricMatrix, or SkewSymMatrix without these methods.
#
# These are the same ownership smell as the `h5save` methods above — `changebackend` is
# `AbstractNeuralNetworks`', the types are `GeometricOptimizers`' — and they belong in a
# `GeometricOptimizers` extension on `AbstractNeuralNetworks`. That is a separate change with its
# own release chain, so they stay here for now.
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
# save / load — dispatch on `NeuralNetwork`, alongside the `NetworkParameters`
# methods in `NeuralNetworkParameters`.
# ---------------------------------------------------------------------------

"""
    save(h5::HDF5.H5DataStore, nn::NeuralNetwork)
    save(filename::AbstractString, nn::NeuralNetwork)

Save the parameters of `nn` to an already-open HDF5 store or to a file.

Extends `save` with a dispatch on `NeuralNetwork`; the parameters themselves are written by
`NeuralNetworkParameters`, which tags each structured leaf with the type to rebuild it as and
records the key order of every group.
"""
save(h5::HDF5.H5DataStore, nn::NeuralNetwork) = save(h5, params(nn))

function save(filename::AbstractString, nn::NeuralNetwork)
    HDF5.h5open(filename, "w") do h5
        save(h5, nn)
    end
    filename
end

"""
    load(::Type{NeuralNetwork}, h5, arch::Architecture; backend = CPU())
    load(::Type{NeuralNetwork}, h5, arch::Architecture, prototype; backend = CPU())

Load parameters from an HDF5 store or file and return a `NeuralNetwork` for `arch`.

The element type is whatever the file holds, so a `Float32` network reloads as `Float32`.

Structured parameters — `StiefelManifold`, `SymmetricMatrix` and the rest — are rebuilt from the
type each was stored under, which `GeometricOptimizers` registers with
`NeuralNetworkParameters.register_parameter_type!`. Pass a `prototype` parameter set of the right
shape to rebuild against it instead and skip the registry altogether.
"""
function load(::Type{NeuralNetwork}, h5::HDF5.H5DataStore, arch::Architecture, args...;
              backend::NeuralNetworkBackend = CPU())
    ps = load(NetworkParameters, h5, args...)
    NeuralNetwork(arch, Chain(arch), ps, backend)
end

function load(::Type{NeuralNetwork}, filename::AbstractString, arch::Architecture, args...;
              backend::NeuralNetworkBackend = CPU())
    HDF5.h5open(filename, "r") do h5
        load(NeuralNetwork, h5, arch, args...; backend = backend)
    end
end

end
