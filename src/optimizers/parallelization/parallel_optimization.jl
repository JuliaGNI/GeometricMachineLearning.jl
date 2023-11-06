@doc raw"""
Parallelizes all *manifold-related computations* involved in the optimization procedure for efficient implementation on GPUs.
"""
mutable struct ParallelOptimization{PT, CT}
    ps::PT
    cache::CT
end

@doc raw"""
Determine the indices where `MultiHeadAttention` layers appear. The inputs are a `Chain` and a `Tuple` (the parameters). 
The outputs are (i) the total count of `MultiHeadAttention` layers appearing and (ii) the indices where they appear in the parameters. 
"""
function determine_multiheadattention_indices(model::Chain, ps::Tuple)
    # count total appearances of MultiHeadAttention layers.
    count = 0
    index_count = 0
    indices = ()
    for (layer, ps_layer) in (model.layers, ps)
        index_count += 1
        if typeof(layer) <: MultiHeadAttention
            count += 1
            indices = (indices..., index_count)
        end
    end
    count, indices
end

function parallelize_weights(model::Chain, ps::Tuple)
    number_of_mha_layers, mha_indices = determine_multiheadattention_indices(model, ps)

    N, n = size(ps[mha_indices[1]].PQ)
end