# The size constants of the reproduction scripts.
#
# A reproduction script states two sizes for every constant that costs time: the size the committed
# result was produced with, and a size that exercises the same code path in seconds. `smoke_size`
# returns the second when `GML_SMOKE` is set in the environment, which is what lets the CI job in
# `.github/workflows/Scripts.yml` prove that a script still runs end to end without training
# anything. A smoke run establishes that the script executes, and nothing about the result.
#
#     include("../utilities/smoke.jl")
#     const N_EPOCHS = smoke_size(2048, 2)

"""
    smoke_size(full, smoke)

Return `smoke` when `GML_SMOKE` is set in the environment and `full` otherwise.
"""
smoke_size(full, smoke) = haskey(ENV, "GML_SMOKE") ? smoke : full
