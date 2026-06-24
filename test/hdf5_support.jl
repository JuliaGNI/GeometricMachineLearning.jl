using GeometricMachineLearning
using Test
using HDF5
import Random

Random.seed!(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ps_eq(a::AbstractArray,    b::AbstractArray)    = a ≈ b
_ps_eq(a::StiefelManifold,  b::StiefelManifold)  = a.A ≈ b.A
_ps_eq(a::SymmetricMatrix,  b::SymmetricMatrix)  = a.S ≈ b.S && a.n == b.n
_ps_eq(a::SkewSymMatrix,    b::SkewSymMatrix)    = a.S ≈ b.S && a.n == b.n
function _ps_eq(a::NamedTuple, b::NamedTuple)
    keys(a) == keys(b) || return false
    all(_ps_eq(a[k], b[k]) for k in keys(a))
end
function _ps_eq(a::NeuralNetworkParameters, b::NeuralNetworkParameters)
    keys(a) == keys(b) || return false
    all(_ps_eq(a[k], b[k]) for k in keys(a))
end

# ---------------------------------------------------------------------------
# save / load roundtrip — one testset per architecture
# ---------------------------------------------------------------------------

# SAE contains PSDLayer → StiefelManifold parameters.
@testset "save/load roundtrip: SymplecticAutoencoder (StiefelManifold)" begin
    arch     = SymplecticAutoencoder(6, 4)
    nn       = NeuralNetwork(arch)
    x        = rand(6)
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
    arch     = SymplecticAutoencoder(6, 4)
    nn       = NeuralNetwork(arch, CPU(), Float32)
    x        = rand(Float32, 6)
    y_before = nn(x)

    mktempdir() do dir
        path = joinpath(dir, "sae_f32.h5")
        save(path, nn)
        nn2 = load(NeuralNetwork, path, arch)

        @test _ps_eq(params(nn), params(nn2))
        @test nn2(x) ≈ y_before
        @test eltype(params(nn2)[1].weight.A) == Float32
    end
end

# save / load also work on an already-open HDF5 store (the lower-level API).
@testset "save/load via open H5DataStore" begin
    arch     = SymplecticAutoencoder(6, 4)
    nn       = NeuralNetwork(arch)
    x        = rand(6)
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

# Smoke test: changebackend applied to a full SAE (CPU → CPU).
@testset "changebackend: full SAE NeuralNetwork (CPU → CPU)" begin
    arch = SymplecticAutoencoder(6, 4)
    nn   = NeuralNetwork(arch)
    nn2  = changebackend(CPU(), nn)
    x    = rand(6)
    @test nn(x) ≈ nn2(x)
end
