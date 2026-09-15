using GeometricMachineLearning
using Test

# Julia only errors on a dangling `export` when the name is actually *resolved*, so an exported name
# that nothing defines is silent: the package loads, the docs build, the suite passes, and
# `using GeometricMachineLearning; ResidualLayer(2)` is an `UndefVarError` for a name the package
# advertises. The allowlist below is empty, so every exported name resolves; an entry is added here
# only with the reason a name is exported and defined nowhere.
#
# The assertions below test what the loaded module defines. `ext/HDF5Ext.jl` loads alongside HDF5
# and this test does not load HDF5, so a name defined only in that extension would read as undefined
# here. No exported name is in that position today.
const UNDEFINED_EXPORTS = Dict{Symbol, String}()

@testset "every exported name is defined" begin
    undefined_exports = filter(
        n -> !isdefined(GeometricMachineLearning, n), names(GeometricMachineLearning))

    # A name may be undefined only if the allowlist above gives a reason for it.
    @test sort(setdiff(undefined_exports, keys(UNDEFINED_EXPORTS))) == Symbol[]

    # And the allowlist may not outlive what it excuses. An entry whose name now resolves, or is no
    # longer exported, has to go; otherwise the next reader trusts a reason that no longer holds.
    @test sort(setdiff(collect(keys(UNDEFINED_EXPORTS)), undefined_exports)) == Symbol[]
end
