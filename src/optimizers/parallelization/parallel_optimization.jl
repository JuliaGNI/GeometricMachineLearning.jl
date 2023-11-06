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

@doc raw"""
This parallelizes all the weights that are on the Stiefel manifold (for `MultiHeadAttentionLayers`).
- the input is a `Chain` (the model) and a `Tuple` that contains the network parameters.
- the output is a tensor that contains all the manifold weights that can be used in further computation.

TODO: Do the allocation with a kernel, this won't run on GPU!
"""
function parallelize_weights(model::Chain, ps::Tuple)
    number_of_mha_layers, mha_indices = determine_multiheadattention_indices(model, ps)

    N, n = size(ps[mha_indices[1]].PQ.head_1.A)
    n_heads = model[mha_indices[1]].n_heads
    # a tensor that stores all the manifold weights
    manifold_tensor = KernelAbstractions.allocate(get_backend(ps[mha_indices[1]].PQ.head_1.A), N, n, 3 * n_heads * number_of_mha_layers)

    total_tensor_index = 0
    # various ``projection types''
    for mha_index in mha_indices
        for PT in (:PQ, :PK, :PV)
            for current_head in 1:n_heads
                head_symbol = Symbol("head_"*string(head))
                total_tensor_index += 1
                manifold_tensor[N, n, total_tensor_index] .= ps[mha_index][PT][head_symbol]
            end
        end
    end
    manifold_tensor
end