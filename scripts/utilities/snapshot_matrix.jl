# The name of the snapshot matrix that `reproduction/symplectic_autoencoders/integration.jl` writes
# and that `training.jl` and `plot_waves.jl` read back.
#
# The name carries the mode because the contents do: a smoke run integrates a 16-site lattice at two
# parameter values where a full run does 128 sites at 20. One name for both would let a full run
# read a smoke-sized matrix that an earlier run had left in the directory, and the `isfile` guards
# in the two readers are what would make that silent rather than loud.

include("smoke.jl")

"""
    snapshot_matrix_file(base = "snapshot_matrix")

Return `\$base.h5` for a full run and `\$(base)_smoke.h5` for a smoke run.
"""
snapshot_matrix_file(base = "snapshot_matrix") = smoke_size("$base.h5", "$(base)_smoke.h5")
