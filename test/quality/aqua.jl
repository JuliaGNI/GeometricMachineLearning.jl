using Aqua
using ExplicitImports
using GeometricMachineLearning
using Test

# Aqua's package-level checks. Seven of the eight `Aqua.test_all` enables by default run; `ambiguities`
# is switched off by name, and the reason is below rather than in the changelog alone.
#
# `Aqua.test_all` wraps each of its checks in a `@testset` of its own and adds no enclosing one, and
# a testset with no parent finalises as soon as it closes -- so called bare, the first failing check
# throws a `TestSetException` and the rest never run. On the suite path the `@safetestset` in
# `runtests.jl` is that parent. The `@testset` below is that parent when this file is run on its
# own, and it is also what lets the count at the end be a further assertion rather than a
# top-level `@test` that aborts the file the moment it fails.
#
# `piracies` runs. The 3-tensor functors of `AbstractNeuralNetworks`' `Dense` and `Linear`, and the
# `save`/`load` methods on `NeuralNetwork`, are `AbstractNeuralNetworks`' own.
#
# `ambiguities` stays off, and the assertion below replaces it. Aqua's check runs in a process of its
# own and asks about whatever that process loaded, so its count moves with the versions of the
# packages a run loads, and with *which* of them a given process had loaded. `detect_ambiguities` on
# this package, in this process, asks only about the methods it defines, and its count is zero. The
# switch stays off rather than becoming a gate on an upstream count that no line of this package
# controls.
@testset "Aqua" begin
    Aqua.test_all(GeometricMachineLearning; ambiguities = false)

    @test isempty(Test.detect_ambiguities(GeometricMachineLearning; recursive = true))
end

# `ExplicitImports` answers a question Aqua does not ask, and the reason to gate on it is not
# tidiness: a `using` or an `import` that nothing needs keeps a dependency reachable from the module
# file, so Aqua's `stale_deps` above sees a package in use that nothing uses. That check only means
# something once these are clean.
#
# Four of the seven checks run here: `no_stale_explicit_imports`, `all_explicit_imports_via_owners`,
# `all_qualified_accesses_via_owners` and `no_self_qualified_accesses`. Two more run in the testset
# below with a list of names, and `no_implicit_imports` is off:
#
#   no_implicit_imports              33 names arrive through a bare `using`, across twelve packages
#                                    -- `norm`, `@kernel`, `rrule`, `pullback` and the rest. Naming
#                                    every one is a change to the whole module header and a judgement
#                                    per name, not a by-product of a dead-code pass.
#
# WHAT THIS GATE DOES NOT CATCH, and one way it can go red on its own.
#
# It reads imports, never definitions. A dead *definition* re-added under `src/` leaves all the
# checks green, so this closes only the import half of the pass that added it.
#
# The two `via_owners` checks ask *which* upstream module defines a name, so they go red when a name
# moves between `GeometricBase`, `GeometricOptimizers` and `AbstractNeuralNetworks` with nothing in
# `src/` having changed. `ExplicitImports = "1.15"` admits any 1.x, so a minor release that sharpens
# stale detection does the same. A red here names the import and the module to take it from, so it
# stays a gate: the fix is one line and the message says which. `test_explicit_imports` opens its
# own `@testset`, so this call needs no wrapper.
#
# `update!` is the one import that nothing in `src/` uses, and it is deliberate: the package does not
# export it, so the import is not there for a re-export, but `GeometricMachineLearning.update!`
# is the binding a caller reaches the `GeometricOptimizers`/`GeometricBase` verb under, and
# `test/quality/exports.jl` asserts it is that object. Nothing here defines a method on it, which is
# what leaves the import with no use to find.
test_explicit_imports(GeometricMachineLearning;
    no_implicit_imports = false,
    no_stale_explicit_imports = (ignore = (:update!,),),
    all_explicit_imports_are_public = false,
    all_qualified_accesses_are_public = false)

# Every name this package reaches in `GeometricOptimizers` is public there, whether it is imported or
# written `GeometricOptimizers.x`. The names below are the ones it reaches in *other* packages that
# those packages do not declare public; each is deliberate, and the fix for each is upstream
# declaring it. No `GeometricOptimizers` name is on either list.
const NON_PUBLIC_QUALIFIED = (
    :Initializer, :dim,                                          # AbstractNeuralNetworks
    :zeros,                                                      # KernelAbstractions
    :jacobian!,                                                  # ForwardDiff
    :JacobianResult, :jacobian, :value)                          # DiffResults
const NON_PUBLIC_IMPORTS = (
    :AbstractExplicitLayer, :Architecture, :_compute_loss, :dim, :model, # AbstractNeuralNetworks
    :load, :save)                                                        # NeuralNetworkParameters

@testset "every name reached in GeometricOptimizers is public" begin
    @test check_all_qualified_accesses_are_public(GeometricMachineLearning;
        ignore = NON_PUBLIC_QUALIFIED) === nothing
    @test check_all_explicit_imports_are_public(GeometricMachineLearning;
        ignore = NON_PUBLIC_IMPORTS) === nothing
end
