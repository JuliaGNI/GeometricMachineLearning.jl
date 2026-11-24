using GeometricMachineLearning
using Test
using HDF5
import Random
import AbstractNeuralNetworks: params
using GeometricMachineLearning: BiorthogonalManifold

Random.seed!(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ps_eq(a::AbstractArray, b::AbstractArray) = a ≈ b
_ps_eq(a::StiefelManifold, b::StiefelManifold) = a.A ≈ b.A
_ps_eq(a::SymmetricMatrix, b::SymmetricMatrix) = a.S ≈ b.S && a.n == b.n
_ps_eq(a::SkewSymMatrix, b::SkewSymMatrix) = a.S ≈ b.S && a.n == b.n
_ps_eq(a::StrictlyLowerTriangular, b::StrictlyLowerTriangular) = a.S ≈ b.S && a.n == b.n
_ps_eq(a::StrictlyUpperTriangular, b::StrictlyUpperTriangular) = a.S ≈ b.S && a.n == b.n
function _ps_eq(a::NamedTuple, b::NamedTuple)
    Set(keys(a)) == Set(keys(b)) || return false
    all(_ps_eq(a[k], b[k]) for k in keys(a))
end
function _ps_eq(a::NetworkParameters, b::NetworkParameters)
    keys(a) == keys(b) || return false
    all(_ps_eq(a[k], b[k]) for k in keys(a))
end

# Reproduces the on-disk layout this package wrote before the traversal moved out: plain nested
# groups with no `kind`/`keys` attributes, and each structured matrix tagged `gml_type`.
function _legacy_group(h5, path)
    path == "/" ? h5 :
    (haskey(h5, path) ? h5[path] : HDF5.create_group(h5, path))
end

function _write_legacy(h5, nt::NamedTuple, path::AbstractString)
    g = _legacy_group(h5, path)
    for (k, v) in pairs(nt)
        _write_legacy(g, v, String(k))
    end
end

_write_legacy(h5, x::AbstractArray, path::AbstractString) = (h5[path] = Array(x); nothing)

function _write_legacy(h5, Y::StiefelManifold, path::AbstractString)
    g = HDF5.create_group(h5, path)
    HDF5.attributes(g)["gml_type"] = "StiefelManifold"
    g["A"] = Array(Y.A)
end

for (T, name) in ((:SymmetricMatrix, "SymmetricMatrix"), (:SkewSymMatrix, "SkewSymMatrix"),
    (:StrictlyLowerTriangular, "StrictlyLowerTriangular"), (
    :StrictlyUpperTriangular, "StrictlyUpperTriangular"))
    @eval function _write_legacy(h5, A::$T, path::AbstractString)
        g = HDF5.create_group(h5, path)
        HDF5.attributes(g)["gml_type"] = $name
        g["S"] = Array(A.S)
        g["n"] = A.n
    end
end

# ---------------------------------------------------------------------------
# save / load roundtrip — one testset per architecture
# ---------------------------------------------------------------------------

# SAE contains PSDLayer → StiefelManifold parameters.
@testset "save/load roundtrip: SymplecticAutoencoder (StiefelManifold)" begin
    arch = SymplecticAutoencoder(10, 4)
    nn = NeuralNetwork(arch)
    x = rand(10)
    y_before = nn(x)

    mktempdir() do dir
        path = joinpath(dir, "sae.h5")
        save(path, nn)
        nn2 = load(NeuralNetwork, path, arch)

        @test _ps_eq(params(nn), params(nn2))
        @test nn2(x) ≈ y_before
    end
end

# Plain Dense layers; the architecture rebuilds the model that the weights are read into.
@testset "save/load roundtrip: StandardAutoencoder" begin
    arch = StandardAutoencoder(
        10, 4; width = 12, n_encoder_layers = 2, n_decoder_layers = 3)
    nn = NeuralNetwork(arch)
    x = rand(10)
    y_before = nn(x)

    mktempdir() do dir
        path = joinpath(dir, "ae.h5")
        save(path, nn)
        nn2 = load(NeuralNetwork, path, arch)

        @test _ps_eq(params(nn), params(nn2))
        @test nn2(x) ≈ y_before
        @test encoder(nn2)(x) ≈ encoder(nn)(x)
    end
end

# LASympNet contains LinearLayer → SymmetricMatrix parameters.
@testset "save/load roundtrip: LASympNet (SymmetricMatrix)" begin
    arch = LASympNet(4)
    nn = NeuralNetwork(arch)
    x = rand(4)
    y_before = nn(x)

    mktempdir() do dir
        path = joinpath(dir, "lasympnet.h5")
        save(path, nn)
        nn2 = load(NeuralNetwork, path, arch)

        @test _ps_eq(params(nn), params(nn2))
        @test nn2(x) ≈ y_before
    end
end

# GSympNet has only plain-array parameters; verify the common path still works.
@testset "save/load roundtrip: GSympNet (plain arrays)" begin
    arch = GSympNet(4)
    nn = NeuralNetwork(arch)
    x = rand(4)
    y_before = nn(x)

    mktempdir() do dir
        path = joinpath(dir, "gsympnet.h5")
        save(path, nn)
        nn2 = load(NeuralNetwork, path, arch)

        @test _ps_eq(params(nn), params(nn2))
        @test nn2(x) ≈ y_before
    end
end

# Float32 roundtrip: GPU training produces Float32 weights.
@testset "save/load roundtrip: Float32 weights (element type preserved)" begin
    arch = SymplecticAutoencoder(10, 4)
    nn = NeuralNetwork(arch, CPU(), Float32)
    x = rand(Float32, 10)
    y_before = nn(x)

    mktempdir() do dir
        path = joinpath(dir, "sae_f32.h5")
        save(path, nn)
        nn2 = load(NeuralNetwork, path, arch)

        @test _ps_eq(params(nn), params(nn2))
        @test nn2(x) ≈ y_before
        # L5 is the first PSDLayer (4 encoder gradient layers precede it); check its StiefelManifold element type.
        @test eltype(params(nn2)[5].weight.A) == Float32
    end
end

# VolumePreservingFeedForward uses StrictlyLowerTriangular / StrictlyUpperTriangular parameters.
@testset "save/load roundtrip: VolumePreservingFeedForward (StrictlyLowerTriangular/StrictlyUpperTriangular)" begin
    arch = VolumePreservingFeedForward(4, 4, 1, tanh)
    nn = NeuralNetwork(arch)
    x = rand(4)
    y_before = nn(x)

    mktempdir() do dir
        path = joinpath(dir, "vpff.h5")
        save(path, nn)
        nn2 = load(NeuralNetwork, path, arch)

        @test _ps_eq(params(nn), params(nn2))
        @test nn2(x) ≈ y_before
    end
end

# save / load also work on an already-open HDF5 store (the lower-level API).
@testset "save/load via open H5DataStore" begin
    arch = SymplecticAutoencoder(10, 4)
    nn = NeuralNetwork(arch)
    x = rand(10)
    y_before = nn(x)

    mktempdir() do dir
        path = joinpath(dir, "sae_store.h5")
        HDF5.h5open(path, "w") do h5
            save(h5, nn)
        end
        nn2 = HDF5.h5open(path, "r") do h5
            load(NeuralNetwork, h5, arch)
        end
        @test nn2(x) ≈ y_before
    end
end

# ---------------------------------------------------------------------------
# Loading against a prototype — the form that consults no registry
# ---------------------------------------------------------------------------

# `load(NeuralNetwork, …, prototype)` rebuilds each structured leaf with `rebuild(prototype_leaf,
# storage)` instead of looking its stored type name up in `NeuralNetworkParameters`' registry. It is
# the path that works for a type nobody registered, so it needs a test of its own rather than riding
# on the roundtrips above, which all go through the registry.
@testset "save/load roundtrip: against a prototype parameter set" begin
    arch = SymplecticAutoencoder(10, 4)
    nn = NeuralNetwork(arch)
    x = rand(10)
    y_before = nn(x)

    # a second network of the same architecture: same shapes, different numbers
    prototype = params(NeuralNetwork(arch))

    mktempdir() do dir
        path = joinpath(dir, "sae_prototype.h5")
        save(path, nn)
        nn2 = load(NeuralNetwork, path, arch, prototype)

        @test _ps_eq(params(nn), params(nn2))
        @test nn2(x) ≈ y_before
        @test params(nn2)[5].weight isa StiefelManifold

        # and on an already-open store
        nn3 = HDF5.h5open(path, "r") do h5
            load(NeuralNetwork, h5, arch, prototype)
        end
        @test nn3(x) ≈ y_before
    end
end

# ---------------------------------------------------------------------------
# LeftInverseSymplecticAutoencoder: one parameter tree for the encoder and the decoder
# ---------------------------------------------------------------------------

# Its model is one `LeftInverseAutoencoderLayer` and not `Chain(arch)`, which the architecture
# refuses; the package's HDF5 extension rebuilds the layer around the parameters read back.
const LEFT_INVERSE = LeftInverseSymplecticAutoencoder(
    6, 2; n_blocks = 2, n_layers = 2, n_ambient_layers = 2)

@testset "save/load roundtrip: LeftInverseSymplecticAutoencoder in $T" for T in (
    Float32, Float64)
    nn = NeuralNetwork(LEFT_INVERSE, CPU(), T)
    z, ξ = rand(T, 6, 5), rand(T, 2, 5)

    mktempdir() do dir
        path = joinpath(dir, "left_inverse.h5")
        save(path, nn)
        nn2 = load(NeuralNetwork, path, LEFT_INVERSE)

        @test nn2.model isa GeometricMachineLearning.LeftInverseAutoencoderLayer
        @test _ps_eq(params(nn), params(nn2))
        @test nn2(z) isa Matrix{T}
        @test encoder(nn2)(z) ≈ encoder(nn)(z)
        @test decoder(nn2)(ξ) ≈ decoder(nn)(ξ)

        # and on an already-open store
        nn3 = HDF5.h5open(path, "r") do h5
            load(NeuralNetwork, h5, LEFT_INVERSE)
        end
        @test nn3.model isa GeometricMachineLearning.LeftInverseAutoencoderLayer
        @test nn3(z) ≈ nn(z)
    end
end

# With biorthogonal lifts, at a point of the manifold drawn by `rand`, so that Φ ≠ Ψ and Φ is not
# orthonormal: the leaf comes back as a `BiorthogonalManifold` with the same entries.
const LEFT_INVERSE_BIORTHOGONAL = LeftInverseSymplecticAutoencoder(
    6, 2; n_blocks = 2, n_layers = 2, n_ambient_layers = 2, biorthogonal = true)

function _biorthogonal_network(::Type{T}) where {T}
    nn = NeuralNetwork(LEFT_INVERSE_BIORTHOGONAL, CPU(), T)
    lifts = [p.weight
             for p in values(params(nn))
             if haskey(p, :weight) && p.weight isa BiorthogonalManifold]
    @test !isempty(lifts)
    foreach(Y -> copyto!(Y, rand(BiorthogonalManifold{T}, size(Y)...)), lifts)
    nn
end

@testset "save/load roundtrip: biorthogonal LeftInverseSymplecticAutoencoder in $T" for T in (
    Float32, Float64)
    nn = _biorthogonal_network(T)
    z, ξ = rand(T, 6, 5), rand(T, 2, 5)

    mktempdir() do dir
        path = joinpath(dir, "left_inverse_biorthogonal.h5")
        save(path, nn)
        nn2 = load(NeuralNetwork, path, LEFT_INVERSE_BIORTHOGONAL)

        @test _ps_eq(params(nn), params(nn2))
        @test any(p -> haskey(p, :weight) && p.weight isa BiorthogonalManifold{T},
            values(params(nn2)))
        @test nn2(z) isa Matrix{T}
        @test encoder(nn2)(z) ≈ encoder(nn)(z)
        @test decoder(nn2)(ξ) ≈ decoder(nn)(ξ)
    end
end

# A file read with an architecture other than the one that saved it fails in `load`, naming the
# parameter, and not at the first call with a bare dimension mismatch.
@testset "load rejects a file of another LeftInverseSymplecticAutoencoder" begin
    re(found, expected) = Regex("^ArgumentError: the parameter `L\\d+\\.weight` in the file is a " *
                                "$found, but the LeftInverseSymplecticAutoencoder expects a $expected; " *
                                "load the file with the architecture it was saved with\\.")
    mktempdir() do dir
        biorthogonal, stiefel = joinpath(dir, "biorthogonal.h5"),
        joinpath(dir, "stiefel.h5")
        save(biorthogonal, _biorthogonal_network(Float64))
        save(stiefel, NeuralNetwork(LEFT_INVERSE))

        @test_throws re("\\d+×\\d+ BiorthogonalManifold", "\\d+×\\d+ StiefelManifold") load(
            NeuralNetwork, biorthogonal, LEFT_INVERSE)
        @test_throws re("\\d+×\\d+ StiefelManifold", "\\d+×\\d+ BiorthogonalManifold") load(
            NeuralNetwork, stiefel, LEFT_INVERSE_BIORTHOGONAL)
        @test_throws r"^ArgumentError: the file holds the layers .*, but the LeftInverseSymplecticAutoencoder has the layers .*; load the file with the architecture it was saved with\." load(
            NeuralNetwork, stiefel, LeftInverseSymplecticAutoencoder(6, 2; n_blocks = 2))
        # as many layers, but the lift one layer earlier than in LEFT_INVERSE, where the file has a
        # gradient layer
        lift_first = LeftInverseSymplecticAutoencoder(
            6, 2; n_blocks = 2, n_layers = 1, n_ambient_layers = 3)
        @test keys(params(NeuralNetwork(lift_first))) ==
              keys(params(NeuralNetwork(LEFT_INVERSE)))
        @test_throws r"^ArgumentError: the layer `L2` in the file has the parameters \(:weight, :bias, :scale\), but the LeftInverseSymplecticAutoencoder expects \(:weight,\); load the file with the architecture it was saved with\." load(
            NeuralNetwork, stiefel, lift_first)

        # the load draws no number from the global generator
        Random.seed!(3)
        a = rand()
        Random.seed!(3)
        load(NeuralNetwork, stiefel, LEFT_INVERSE)
        @test rand() == a
    end
end

# Renames the stored type of every structured leaf under `g` to one that no package registered, so
# that only a load against a prototype can rebuild the leaves; returns the number of leaves renamed.
function _unregister_leaf_types!(g)
    n = 0
    for k in keys(g)
        c = g[k]
        c isa HDF5.Group || continue
        if haskey(HDF5.attributes(c), "parameter_type")
            HDF5.delete_attribute(c, "parameter_type")
            HDF5.attributes(c)["parameter_type"] = "UnregisteredLeaf"
            n += 1
        end
        n += _unregister_leaf_types!(c)
    end
    n
end

# The prototype reaches the parameters: the file names a leaf type that the registry does not know,
# so the load without a prototype fails, and the load against one rebuilds the Stiefel leaves.
@testset "save/load roundtrip: LeftInverseSymplecticAutoencoder against a prototype" begin
    nn = NeuralNetwork(LEFT_INVERSE)
    z = rand(6, 5)
    prototype = params(NeuralNetwork(LEFT_INVERSE))

    mktempdir() do dir
        path = joinpath(dir, "left_inverse_prototype.h5")
        save(path, nn)
        renamed = HDF5.h5open(_unregister_leaf_types!, path, "r+")
        @test renamed > 0
        @test_throws r"^ArgumentError: the file contains a parameter of type `UnregisteredLeaf`, which is not registered\." load(
            NeuralNetwork, path, LEFT_INVERSE)

        nn2 = load(NeuralNetwork, path, LEFT_INVERSE, prototype)
        @test nn2.model isa GeometricMachineLearning.LeftInverseAutoencoderLayer
        @test _ps_eq(params(nn), params(nn2))
        @test nn2(z) ≈ nn(z)

        nn3 = HDF5.h5open(path, "r") do h5
            load(NeuralNetwork, h5, LEFT_INVERSE, prototype)
        end
        @test nn3(z) ≈ nn(z)
    end
end

# ---------------------------------------------------------------------------
# Files written before the traversal moved out of this package
# ---------------------------------------------------------------------------

# This package used to write each structured matrix itself, as a group tagged `gml_type` holding the
# fields under their own names and recording no key order. `NeuralNetworkParameters` recognises the
# tag and rebuilds through the registry `GeometricOptimizers` fills, so those files still load —
# which is the whole reason the duplicated reader here could be deleted rather than kept alongside.
# `SymmetricMatrix` and `StiefelManifold` are the two shapes the old writer produced, and
# `GeometricOptimizers` normalises them through different helpers — `S`/`n` for a storage matrix,
# a bare `A` for a manifold element — so both legs need reading back.
@testset "a file in the old gml_type layout still loads" begin
    for (name, arch, dimin) in (("LASympNet (SymmetricMatrix)", LASympNet(4), 4),
        ("SymplecticAutoencoder (StiefelManifold)",
        SymplecticAutoencoder(10, 4), 10))
        @testset "$name" begin
            nn = NeuralNetwork(arch)
            ps = params(nn)
            x = rand(dimin)
            y = nn(x)

            mktempdir() do dir
                path = joinpath(dir, "legacy.h5")
                HDF5.h5open(path, "w") do h5
                    _write_legacy(h5, params(ps), "/")
                end
                nn2 = load(NeuralNetwork, path, arch)

                @test keys(params(nn2)) == keys(ps)
                @test _ps_eq(ps, params(nn2))
                @test nn2(x) ≈ y
            end
        end
    end
end
