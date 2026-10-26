using GeometricMachineLearning
using Test

# Julia only errors on a dangling `export` when the name is actually *resolved*, so an exported name
# that nothing defines is silent: the package loads, the docs build, the suite passes, and
# `using GeometricMachineLearning; ResidualLayer(2)` is an `UndefVarError` for a name the package
# advertises. The allowlist below is empty, so every exported name resolves; an entry is added here
# only with the reason a name is exported and defined nowhere.
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

# The names this package does not hand to `using GeometricMachineLearning`: `update!` and `solve!`
# are `GeometricOptimizers`'/`SimpleSolvers`' verbs, `iterate` is `Base`'s, and `PoissonTensor` is
# gone. The first three stay bound -- `solve!` carries this package's methods, while `update!` and
# `iterate` are their owners' objects -- so a test for them is a test on the export and not on the
# definition.
const NOT_EXPORTED = (:update!, :solve!, :iterate, :PoissonTensor)

@testset "the four names are not exported" begin
    for name in NOT_EXPORTED
        @test !Base.isexported(GeometricMachineLearning, name)
        @test !(name in names(GeometricMachineLearning))
    end
    # The two verbs are still bound here, as the owners' objects.
    @test nameof(parentmodule(GeometricMachineLearning.update!)) === :GeometricBase
    @test nameof(parentmodule(GeometricMachineLearning.solve!)) === :SimpleSolvers
    @test !isdefined(GeometricMachineLearning, :PoissonTensor)
end

# A caller who loads both packages keeps every name of both: each name the two export is one binding,
# so neither `using` shadows the other, and `Optimizer` is `GeometricOptimizers`' solver.
@testset "using GeometricOptimizers and GeometricMachineLearning together" begin
    m = Module()
    Core.eval(m, :(using GeometricOptimizers, GeometricMachineLearning))
    @test Core.eval(m, :(Optimizer === GeometricOptimizers.Optimizer))

    GO = Core.eval(m, :GeometricOptimizers)
    shared = intersect(names(GO), names(GeometricMachineLearning))
    @test :Adam in shared
    @test filter(n -> getfield(GO, n) !== getfield(GeometricMachineLearning, n), shared) ==
          Symbol[]

    # no alias names a method under a second name
    for alias in (:GradientOptimizer, :MomentumOptimizer, :AdamOptimizer)
        @test !isdefined(GeometricMachineLearning, alias)
    end
end
