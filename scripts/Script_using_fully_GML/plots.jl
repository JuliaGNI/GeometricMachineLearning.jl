# `plot_hnn` and `plot_network_sim` are defined once, in `scripts/plots.jl`, and pulled in here so
# that the scripts in this directory get them under the name they expect.
include(joinpath(@__DIR__, "..", "plots.jl"))
