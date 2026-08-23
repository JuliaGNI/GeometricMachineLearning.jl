using GeometricMachineLearning
using Test
using HDF5
using LinearAlgebra: qr
import Random
import AbstractNeuralNetworks: params, changebackend

Random.seed!(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ps_eq(a::AbstractArray,    b::AbstractArray)    = a ≈ b
_ps_eq(a::StiefelManifold,  b::StiefelManifold)  = a.A ≈ b.A
_ps_eq(a::SymmetricMatrix,  b::SymmetricMatrix)  = a.S ≈ b.S && a.n == b.n
_ps_eq(a::SkewSymMatrix,    b::SkewSymMatrix)    = a.S ≈ b.S && a.n == b.n
_ps_eq(a::LowerTriangular,  b::LowerTriangular)  = a.S ≈ b.S && a.n == b.n
_ps_eq(a::UpperTriangular,  b::UpperTriangular)  = a.S ≈ b.S && a.n == b.n
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
_legacy_group(h5, path) = path == "/" ? h5 :
                          (haskey(h5, path) ? h5[path] : HDF5.create_group(h5, path))

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
                  (:LowerTriangular, "LowerTriangular"), (:UpperTriangular, "UpperTriangular"))
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
    arch     = SymplecticAutoencoder(10, 4)
    nn       = NeuralNetwork(arch)
    x        = rand(10)
    y_before = nn(x)

    mktempdir() do dir
        path = joinpath(dir, "sae.h5")
        save(path, nn)
        nn2 = load(NeuralNetwork, path, arch)

        @test _ps_eq(params(nn), params(nn2))
        @test nn2(x) ≈ y_before
    end
end

# LASympNet contains LinearLayer → SymmetricMatrix parameters.
@testset "save/load roundtrip: LASympNet (SymmetricMatrix)" begin
    arch     = LASympNet(4)
    nn       = NeuralNetwork(arch)
    x        = rand(4)
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
    arch     = GSympNet(4)
    nn       = NeuralNetwork(arch)
    x        = rand(4)
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
    arch     = SymplecticAutoencoder(10, 4)
    nn       = NeuralNetwork(arch, CPU(), Float32)
    x        = rand(Float32, 10)
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

# VolumePreservingFeedForward uses LowerTriangular / UpperTriangular parameters.
@testset "save/load roundtrip: VolumePreservingFeedForward (LowerTriangular/UpperTriangular)" begin
    arch     = VolumePreservingFeedForward(4, 4, 1, tanh)
    nn       = NeuralNetwork(arch)
    x        = rand(4)
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
    arch     = SymplecticAutoencoder(10, 4)
    nn       = NeuralNetwork(arch)
    x        = rand(10)
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
    arch     = SymplecticAutoencoder(10, 4)
    nn       = NeuralNetwork(arch)
    x        = rand(10)
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
            x  = rand(dimin)
            y  = nn(x)

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

# ---------------------------------------------------------------------------
# changebackend — new methods for GML special array types
# ---------------------------------------------------------------------------

@testset "changebackend: StiefelManifold (CPU → CPU)" begin
    Y  = StiefelManifold(Matrix(qr(randn(6, 4)).Q))
    Y2 = changebackend(CPU(), Y)
    @test Y2 isa StiefelManifold
    @test Y.A ≈ Y2.A
end

@testset "changebackend: SymmetricMatrix (CPU → CPU)" begin
    A  = SymmetricMatrix(rand(10), 4)
    A2 = changebackend(CPU(), A)
    @test A2 isa SymmetricMatrix
    @test A.S ≈ A2.S
    @test A.n == A2.n
end

@testset "changebackend: SkewSymMatrix (CPU → CPU)" begin
    A  = SkewSymMatrix(rand(6), 4)
    A2 = changebackend(CPU(), A)
    @test A2 isa SkewSymMatrix
    @test A.S ≈ A2.S
    @test A.n == A2.n
end

@testset "changebackend: LowerTriangular (CPU → CPU)" begin
    A  = LowerTriangular(rand(6), 4)
    A2 = changebackend(CPU(), A)
    @test A2 isa LowerTriangular
    @test A.S ≈ A2.S
    @test A.n == A2.n
end

@testset "changebackend: UpperTriangular (CPU → CPU)" begin
    A  = UpperTriangular(rand(6), 4)
    A2 = changebackend(CPU(), A)
    @test A2 isa UpperTriangular
    @test A.S ≈ A2.S
    @test A.n == A2.n
end

# Smoke test: changebackend applied to a full SAE (CPU → CPU).
@testset "changebackend: full SAE NeuralNetwork (CPU → CPU)" begin
    arch = SymplecticAutoencoder(10, 4)
    nn   = NeuralNetwork(arch)
    nn2  = changebackend(CPU(), nn)
    x    = rand(10)
    @test nn(x) ≈ nn2(x)
end
