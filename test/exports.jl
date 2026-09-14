using GeometricMachineLearning
using Test

# Julia only errors on a dangling `export` when the name is actually *resolved*, so an exported name
# that nothing defines is silent: the package loads, the docs build, the suite passes, and
# `using GeometricMachineLearning; ResidualLayer(2)` is an `UndefVarError` for a name the package
# advertises. Ten such names are in the export list today; each is allowlisted below with its export
# site and the reason it resolves to nothing, and each needs a decision -- define it, or drop the
# export. See *C10* under *Open Issues* in the changelog. Until then this test stops an eleventh
# from being added.
#
# "No file under `src/` defines it" is what the assertion below establishes: `src/` is all the
# package loads, so a name it left undefined is defined in none of it.
const UNDEFINED_EXPORTS = Dict(
    # Exported together under the comment "GPU specific operations". `test/performance_tests/`
    # calls all three; *C11* records that tree as unreachable from `runtests.jl`.
    :convert_to_dev => "exported at `GeometricMachineLearning.jl:182`; no file under `src/` defines it",
    :Device => "exported at `GeometricMachineLearning.jl:182`; no file under `src/` defines it",
    :CPUDevice => "exported at `GeometricMachineLearning.jl:182`; no file under `src/` defines it",

    # `ResidualLayer` is the one name here that has a definition; it just is not loaded.
    :ResidualLayer => "exported at `GeometricMachineLearning.jl:187`; its definition is at `legacy/layers/resnet.jl:6`, which nothing under `src/` includes. The loaded layer of that shape is `ResNetLayer` (`src/layers/resnet.jl:17`)",
    :LinearSymplecticLayerP => "exported at `GeometricMachineLearning.jl:188`; no file under `src/` defines it",
    :LinearSymplecticLayerQ => "exported at `GeometricMachineLearning.jl:188`; no file under `src/` defines it",

    # Two generics `GeometricBase` defines but does not *export*, so `using GeometricBase` does not
    # bring them into scope and an explicit `import` is needed.
    :description => "exported at `GeometricMachineLearning.jl:114`; `GeometricBase` defines `description` but does not export it, so the `import` is missing",
    :timestep => "exported at `GeometricMachineLearning.jl:374`; `GeometricBase` defines `timestep` but does not export it, so the `import` is missing",

    # Two names in the `train!` subsystem's export blocks.
    :symbol => "exported at `GeometricMachineLearning.jl:281`; no file under `src/` defines it",
    :aresame => "exported at `GeometricMachineLearning.jl:291`; no file under `src/` defines it"
)

@testset "every exported name is defined" begin
    undefined_exports = filter(
        n -> !isdefined(GeometricMachineLearning, n), names(GeometricMachineLearning))

    # A name may be undefined only if the allowlist above gives a reason for it.
    @test sort(setdiff(undefined_exports, keys(UNDEFINED_EXPORTS))) == Symbol[]
end
