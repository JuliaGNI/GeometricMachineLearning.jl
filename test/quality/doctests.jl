# The doctests of this package, in its docstrings and in the manual under `docs/src`, as the
# `Doctests` job of `CI.yml` runs them.
#
# Documenter evaluates a page's `@meta` block in `Main`, and a `@safetestset` file runs in a module
# of its own, so `GeometricMachineLearning` is imported into `Main` first.

using GeometricMachineLearning
using Documenter: DocMeta, doctest

@eval Main import GeometricMachineLearning

DocMeta.setdocmeta!(GeometricMachineLearning, :DocTestSetup,
    :(using GeometricMachineLearning); recursive = true)

doctest(GeometricMachineLearning)
