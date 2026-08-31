# Convert pre-trained parameter files from JLD2 to HDF5.
#
# Run from the repo root:
#   julia scripts/convert_jld2_to_h5.jl
#
# The script activates the docs project (which has JLD2, HDF5, and GML) and
# writes one .h5 file per neural network stored in each .jld2 source file.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"))
Pkg.develop(path = joinpath(@__DIR__, ".."); io = devnull)

using JLD2, GeometricMachineLearning, HDF5
import AbstractNeuralNetworks: h5save

# Convert a single Tuple of layer-NamedTuples to an HDF5 file.
# The HDF5 structure mirrors what GML's save(filename, nn) produces:
#   /L1/...  /L2/...  etc.
# Special GML types (StiefelManifold, SymmetricMatrix, SkewSymMatrix) are
# serialised with a gml_type attribute so that load(NeuralNetwork, ...) can
# reconstruct them.
function jld2_to_h5(src_file::String, src_key::String, dst_file::String)
    ps = JLD2.load(src_file)[src_key]   # Tuple{NamedTuple, ...}
    n = length(ps)
    nt = NamedTuple{Tuple(Symbol("L$i") for i in 1:n)}(ps)
    HDF5.h5open(dst_file, "w") do h5
        h5save(h5, nt, "/")
    end
    println("  $src_file [\"$src_key\"]  →  $dst_file  ($n layers)")
end

const T = joinpath(@__DIR__, "..", "docs", "src", "tutorials")

println("=== Symplectic autoencoder ===")
jld2_to_h5("$T/sae_parameters.jld2", "sae_parameters", "$T/sae_parameters.h5")
jld2_to_h5("$T/integrator_parameters.jld2", "integrator_parameters", "$T/integrator_parameters.h5")
jld2_to_h5("$T/integrator_parameters_psd.jld2", "integrator_parameters", "$T/integrator_parameters_psd.h5")

println("\n=== Volume-preserving transformer (rigid body) ===")
jld2_to_h5("$T/transformer_rigid_body.jld2", "nn_vpff_params", "$T/transformer_rigid_body_nn_vpff.h5")
jld2_to_h5("$T/transformer_rigid_body.jld2", "nn_vpt_arb_params", "$T/transformer_rigid_body_nn_vpt.h5")
jld2_to_h5("$T/transformer_rigid_body.jld2", "nn_st_params", "$T/transformer_rigid_body_nn_st.h5")

println("\nDone.")
