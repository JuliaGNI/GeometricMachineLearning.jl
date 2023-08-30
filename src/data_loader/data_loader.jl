"""
Data Loader is a struct that creates an instance based on a tensor (or different input format) and is designed to make training convenient.

Implemented: 
If the data loader is called with a single tensor, a batch_size and an output_size, then the batch is drawn randomly in the relevant range and the output is assigned accordingly.

TODO: Implement DataLoader that works well with GeometricEnsembles etc. 
"""
struct DataLoader{T, AT<:AbstractArray{T}}
    data::AT
    batch::AT
    output::AT
    sys_dim::Integer
    seq_length::Integer 
    batch_size::Integer
    output_size::Integer 
    n_params::Integer 
    n_time_steps::Integer 
end

function DataLoader(data::AbstractArray{T, 3}, seq_length=10, batch_size=32, output_size=1) where T
    @info "You have provided a tensor with three axes as input. They will be interpreted as \n (i) system dimension, (ii) number of parameters and (iii) number of time steps."
    sys_dim,n_params,n_time_steps = size(data)
    backend = KernelAbstractions.get_backend(data)
    batch = KernelAbstractions.allocate(backend, T, sys_dim, seq_length, batch_size)
    output = KernelAbstractions.allocate(backend, T, sys_dim, output_size, batch_size)
    draw_batch!(batch, output, data, seq_length, batch_size, output_size, n_params, n_time_steps)
    DataLoader{T, typeof(data)}(data, batch, output, sys_dim, seq_length, batch_size, output_size, n_params, n_time_steps)
end

function redraw_batch(dl::DataLoader)
    draw_batch!(dl.batch, dl.output, dl.data, dl.seq_length, dl.batch_size, dl.output_size, dl.n_params, dl.n_time_steps)
end

function loss(model::Union{Chain, AbstractExplicitLayer}, ps::Union{Tuple, NamedTuple}, dl::DataLoader{T}) where T
    batch_output = model(dl.batch, ps)
    output_estimate = assign_output_estimate(batch_output, dl.output_size)
    norm(dl.output - output_estimate)/T(sqrt(dl.batch_size))/T(sqrt(dl.output_size))
end