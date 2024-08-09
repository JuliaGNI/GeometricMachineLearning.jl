# This assigns the batch if the data are in form of a matrix.
function draw_batch!(batch::AbstractMatrix{T}, data::AbstractMatrix{T}) where T
    sys_dim, batch_size = size(batch)
    n_params = size(data, 2)
    param_indices = Int.(ceil.(rand(T, batch_size)*n_params))
    batch .= data[:, param_indices]
end