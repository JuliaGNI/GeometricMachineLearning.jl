# Assigning a batch and its target out of the full data tensor is `src/data_loader/batch.jl`'s job:
# its gather kernel reads a matrix of `(time_step, parameter)` indices. This file crops the
# estimated output instead.

@doc raw"""
    assign_output_estimate(full_output, prediction_window)

Crop the output to get the correct number of output vectors.

The function `assign_output_estimate` is closely related to the [`Transformer`](@ref). 
It takes the last `prediction_window` columns of the output and uses them for the prediction.

i.e.
```math
\mathbb{R}^{N\times{}T}\to\mathbb{R}^{N\times\mathtt{pw}}, 
\begin{bmatrix} 
    z^{(1)}_1               & \cdots & z^{(T)}_1 \\ 
    \cdots                  & \cdots & \cdots    \\ 
    z^{(1)}_n               & \cdots & z^{(T)}_n
    \end{bmatrix} \mapsto 
    \begin{bmatrix} 
    z^{(T - \mathtt{pw} + 1)}_1 & \cdots      & z^{(T)}_1 \\ 
    \cdots                  & \cdots      & \cdots \\ 
    z^{(T - \mathtt{pw} + 1)}_n & \cdots      & z^{(T)}_n\end{bmatrix}     
``` 

If `prediction_window` is equal to `sequence_length`, then this is not needed.
"""
function assign_output_estimate(
        full_output::AbstractArray{T, 3}, prediction_window::Int) where {T}
    full_output[:, (end - prediction_window + 1):end, :]
end
