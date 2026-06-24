using HDF5
import AbstractNeuralNetworks: h5save, changebackend, NeuralNetworkBackend, save, load

# ---------------------------------------------------------------------------
# h5save — new methods for GML special array types
#
# AbstractNeuralNetworks only defines h5save for AbstractArray and NamedTuple.
# PSDLayer stores StiefelManifold and LASympNet's LinearLayer stores
# SymmetricMatrix; without these methods h5save throws a MethodError on any
# network whose parameters include those types.
# ---------------------------------------------------------------------------

function h5save(h5::HDF5.H5DataStore, Y::StiefelManifold, path::AbstractString)
    group = haskey(h5, path) ? h5[path] : HDF5.create_group(h5, path)
    HDF5.attributes(group)["gml_type"] = "StiefelManifold"
    group["A"] = Array(Y.A)
end

function h5save(h5::HDF5.H5DataStore, A::SymmetricMatrix, path::AbstractString)
    group = haskey(h5, path) ? h5[path] : HDF5.create_group(h5, path)
    HDF5.attributes(group)["gml_type"] = "SymmetricMatrix"
    group["S"] = Array(A.S)
    group["n"] = A.n
end

function h5save(h5::HDF5.H5DataStore, A::SkewSymMatrix, path::AbstractString)
    group = haskey(h5, path) ? h5[path] : HDF5.create_group(h5, path)
    HDF5.attributes(group)["gml_type"] = "SkewSymMatrix"
    group["S"] = Array(A.S)
    group["n"] = A.n
end

# ---------------------------------------------------------------------------
# changebackend — new methods for GML special array types
#
# AbstractNeuralNetworks.changebackend handles AbstractArray and NamedTuple.
# Moving a NeuralNetwork between devices fails for parameters that include
# StiefelManifold, SymmetricMatrix, or SkewSymMatrix without these methods.
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

# ---------------------------------------------------------------------------
# Internal recursive loader that reconstructs GML special types from the
# gml_type attribute written by h5save.  Kept private; used only by the
# load methods below so that we do not shadow AbstractNeuralNetworks.h5load.
# ---------------------------------------------------------------------------

_gml_h5load(ds::HDF5.Dataset) = read(ds)

function _gml_h5load(group::HDF5.Group)
    if haskey(HDF5.attributes(group), "gml_type")
        gml_type = read(HDF5.attributes(group)["gml_type"])
        if gml_type == "StiefelManifold"
            return StiefelManifold(read(group["A"]))
        elseif gml_type == "SymmetricMatrix"
            return SymmetricMatrix(read(group["S"]), read(group["n"]))
        elseif gml_type == "SkewSymMatrix"
            return SkewSymMatrix(read(group["S"]), read(group["n"]))
        end
    end
    paramkeys = Tuple(Symbol.(keys(group)))
    paramvals = Tuple(_gml_h5load(group[k]) for k in keys(group))
    NamedTuple{paramkeys}(paramvals)
end

# ---------------------------------------------------------------------------
# save — new dispatch on NeuralNetwork, mirroring the existing
#   save(h5::H5DataStore, p::NeuralNetworkParameters)
# method in AbstractNeuralNetworks.
# ---------------------------------------------------------------------------

"""
    save(h5::HDF5.H5DataStore, nn::NeuralNetwork)

Save the parameters of `nn` into an already-open HDF5 store.

Extends `AbstractNeuralNetworks.save` with a dispatch on `NeuralNetwork`.
GML special array types (`StiefelManifold`, `SymmetricMatrix`, `SkewSymMatrix`)
are tagged with a `gml_type` attribute so that [`load`](@ref) can reconstruct
them faithfully.
"""
function save(h5::HDF5.H5DataStore, nn::NeuralNetwork)
    h5save(h5, AbstractNeuralNetworks.params(params(nn)), "/")
end

"""
    save(filename::AbstractString, nn::NeuralNetwork)

Convenience overload: open `filename` for writing, then call
`save(h5, nn)`.
"""
function save(filename::AbstractString, nn::NeuralNetwork)
    HDF5.h5open(filename, "w") do h5
        save(h5, nn)
    end
end

# ---------------------------------------------------------------------------
# load — new dispatch on NeuralNetwork, mirroring the existing
#   load(::Type{NeuralNetworkParameters}, h5::H5DataStore)
# method in AbstractNeuralNetworks.
# ---------------------------------------------------------------------------

"""
    load(::Type{NeuralNetwork}, h5::HDF5.H5DataStore, arch::Architecture; backend=CPU())

Load network parameters from an already-open HDF5 store and return a
`NeuralNetwork` for `arch`.

Extends `AbstractNeuralNetworks.load` with a dispatch on `NeuralNetwork`.
GML special array types are reconstructed from their `gml_type` attribute.
The element type is preserved as stored (Float32 files reload as Float32).
"""
function load(::Type{NeuralNetwork}, h5::HDF5.H5DataStore, arch::Architecture;
              backend::NeuralNetworkBackend = CPU())
    ps = NeuralNetworkParameters(_gml_h5load(h5["/"]))
    NeuralNetwork(arch, Chain(arch), ps, backend)
end

"""
    load(::Type{NeuralNetwork}, filename::AbstractString, arch::Architecture; backend=CPU())

Convenience overload: open `filename` for reading, then call
`load(NeuralNetwork, h5, arch; backend)`.
"""
function load(::Type{NeuralNetwork}, filename::AbstractString, arch::Architecture;
              backend::NeuralNetworkBackend = CPU())
    HDF5.h5open(filename, "r") do h5
        load(NeuralNetwork, h5, arch; backend = backend)
    end
end
