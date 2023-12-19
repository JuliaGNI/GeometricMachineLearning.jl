@doc raw"""
Data Loader is a struct that creates an instance based on a tensor (or different input format) and is designed to make training convenient. 

The fields of the struct are the following: 
- `data`: The input data with axes (i) system dimension, (ii) number of parameters and (iii) number of time steps.
- `output`: The tensor that contains the output (supervised learning) - this may be of type Nothing if the constructor is only called with one tensor (unsupervised learning).
- `input_dim`: The *dimension* of the system, i.e. what is taken as input by a regular neural network.
- `input_time_steps`: The length of the entire time series of the data
- `n_params`: The number of parameters that are present in the data set (length of third axis)
- `output_dim`: The dimension of the output tensor (first axis). 
- `output_time_steps`: The size of the second axis of the output tensor (also called prediction_window, `output_time_steps=1` in most cases)

If for the output we have a tensor whose second axis has length 1, we still store it as a tensor and not a matrix. This is because it is not necessarily of length 1. 

TODO: Implement DataLoader that works well with EnsembleProblems etc.
"""
struct DataLoader{T, AT<:Union{NamedTuple, AbstractArray{T}}, OT<:Union{AbstractArray, Nothing}}
    input::AT
    output::OT
    input_dim::Integer
    input_time_steps::Union{Integer, Nothing}
    n_params::Union{Integer, Nothing} 
    output_dim::Union{Integer, Nothing}
    output_time_steps::Union{Integer, Nothing}
end

"""
The `DataLoader` is called with a single tensor (**snapshot tensor**)
"""
function DataLoader(data::AbstractArray{T, 3}) where T
    @info "You have provided a tensor with three axes as input. They will be interpreted as \n (i) system dimension, (ii) number of time steps and (iii) number of params."
    input_dim, input_time_steps, n_params = size(data)
    DataLoader{T, typeof(data), Nothing}(data, nothing, input_dim, input_time_steps, n_params, nothing, nothing)
end

"""
The DataLoader is called with a single matrix (**snapshot matrix**)
"""
function DataLoader(data::AbstractMatrix{T}) where T 
    @info "You have provided a matrix as input. The axes will be interpreted as (i) system dimension and (ii) number of parameters."
    input_dim, n_params = size(data)
    DataLoader{T, typeof(data), Nothing}(data, nothing, input_dim, nothing, n_params, nothing, nothing)
end

# T and T1 are not the same because T1 is of Integer type
"""
The DataLoader is called with a tensor and a vector. For the moment this is always interpreted to be the MNIST data set. 
"""
function DataLoader(data::AbstractArray{T, 3}, target::AbstractVector{T1}; patch_length=7) where {T, T1} 
    @info "You provided a tensor and a vector as input. This will be treated as a classification problem (MNIST). Tensor axes: (i) & (ii) image axes and (iii) parameter dimension."
    im_dim₁, im_dim₂, n_params = size(data)
    @assert length(target) == n_params 
    number_of_patches = (im_dim₁÷patch_length)*(im_dim₂÷patch_length) 
    target = onehotbatch(target)
    data_preprocessed = split_and_flatten(data, patch_length=patch_length, number_of_patches=number_of_patches)
    DataLoader{T, typeof(data_preprocessed), typeof(target)}(
        data_preprocessed, target, patch_length^2, number_of_patches, n_params, 10, 1
        )
end

@doc raw"""
`DataLoader` for `NamedTuple` that has `q` and `p` as keys.

Here the dimension of the `DataLoader` (`input_dim`) is interpreted as the $q$- and $p$-dimension combined, i.e. if $q$ and $p$ both evolve on $\mathbb{R}^n$, then the dimension of the instance of `DataLoader` is $2n$.

Here the number of time steps is the *length of the second axis* of the input minus one. This means that for $(z^{(0)}, \ldots, z^{(T)})$ `input_time_steps=T`.

TODO: implement the autocoder setting *in a good way*.
"""
function DataLoader(data::NamedTuple{(:q, :p), Tuple{AT, AT}}) where {T, AT<:AbstractMatrix{T}}
    @info "You have provided a NamedTuple with keys q and p; the data are matrices. This is interpreted as *symplectic data*."
    
    dim2, time_steps = size(data.q)
    DataLoader{T, typeof(data), Nothing}(data, nothing, dim2*2, time_steps-1, nothing, nothing, nothing)
end

function DataLoader(data::NamedTuple{(:q, :p), Tuple{AT, AT}}; output_time_steps=1) where {T, AT<:AbstractArray{T, 3}}
    @info "You have provided a NamedTuple with keys q and p; the data are tensors. This is interpreted as *symplectic data*."
    
    dim2, time_steps, n_params = size(data.q)
    DataLoader{T, typeof(data), Nothing}(data, nothing, dim2*2, time_steps-1, n_params, nothing, output_time_steps)
end

@doc raw"""
Computes the loss for a neural network and a data set. 
The computed loss is $||output - \mathcal{NN}(input)||_F/\mathtt{size(output, 2)}/\mathtt{size(output, 3)}$, where $||A||_F := \sqrt{\sum_{i_1,\ldots,i_k}||a_{i_1,\ldots,i_k}^2}$ is the Frobenius norm.

It takes as input: 
- `model`
- `ps`
- `input`
- `output`
"""
function loss(model::Union{Chain, AbstractExplicitLayer}, ps::Union{Tuple, NamedTuple}, input::AT, output::BT) where {T, T1, AT<:AbstractArray{T, 3}, BT<:AbstractArray{T1, 3}}
    output_estimate = model(input, ps)
    norm(output - output_estimate) / norm(output) # /T(sqrt(size(output, 2)*size(output, 3)))
end

@doc raw"""
The *autoencoder loss*. 
"""
function loss(model::Chain, ps::Tuple, input::BT) where {T, BT<:AbstractArray{T}} 
    output_estimate = model(input, ps)
    norm(output_estimate - input) / norm(input) # /T(sqrt(size(input, 2)*size(input, 3)))
end

nt_diff(A, B) = (q = A.q - B.q, p = A.p - B.p)
nt_norm(A) = norm(A.q) + norm(A.p)

function loss(model::Chain, ps::Tuple, input::NT) where {T, AT<:AbstractArray{T}, NT<:NamedTuple{(:q, :p,), Tuple{AT, AT}}}
    output_estimate = model(input, ps)
    nt_norm(nt_diff(output_estimate, input)) / nt_norm(input)
end

@doc raw"""
Loss function that takes a `NamedTuple` as input. This should be used with a SympNet (or other neural network-based integrator). It computes:

```math
\mathtt{loss}(\mathcal{NN}, \mathtt{ps}, \begin{pmatrix} q \\ p \end{pmatrix}, \begin{pmatrix} q' \\ p' \end{pmatrix}) \mapsto \left|| \mathcal{NN}(\begin{pmatrix} q \\ p \end{pmatrix}) -  \begin{pmatrix} q' \\ p' \end{pmatrix} \right|| / \left|| \begin{pmatrix} q \\ p \end{pmatrix} \right||
```
"""
function loss(model::Chain, ps::Tuple, input::NamedTuple, output::NamedTuple) 
    output_estimate = model(input, ps)
    nt_norm(nt_diff(output_estimate, output)) / nt_norm(input)
end

@doc raw"""
Alternative call of the loss function. This takes as input: 
- `model`
- `ps`
- `dl::DataLoader`
"""
function loss(model::Union{Chain, AbstractExplicitLayer}, ps::Union{Tuple, NamedTuple}, dl::DataLoader{T, AT, BT}) where {T, T1, AT<:AbstractArray{T, 3}, BT<:AbstractArray{T1, 3}}
    loss(model, ps, dl.input, dl.output)
end

function loss(model::Chain, ps::Tuple, dl::DataLoader{T, BT, Nothing}) where {T, BT<:AbstractArray{T, 3}} 
    loss(model, ps, dl.input)
end

function loss(model::Chain, ps::Tuple, dl::DataLoader{T, BT, Nothing}) where {T, BT<:AbstractArray{T, 2}} 
    loss(model, ps, dl.input)
end

function loss(model::Chain, ps::Tuple, dl::DataLoader{T, BT}) where {T, BT<:NamedTuple}
    loss(model, ps, dl.input)
end

@doc raw"""
Wrapper if we deal with a neural network.
"""
function loss(nn::NeuralNetwork, dl::DataLoader)
    loss(nn.model, nn.params, dl)
end

@doc raw"""
Computes the accuracy (as opposed to the loss) of a neural network classifier. 

It takes as input:
- `model::Chain`:
- `ps`: parameters of the network
- `dl::DataLoader`
"""
function accuracy(model::Chain, ps::Tuple, dl::DataLoader{T, AT, BT}) where {T, T1<:Integer, AT<:AbstractArray{T}, BT<:AbstractArray{T1}}
    output_tensor = model(dl.input, ps)
    output_estimate = assign_output_estimate(output_tensor, dl.output_time_steps)
    backend = KernelAbstractions.get_backend(output_estimate)
    tensor_of_maximum_elements = KernelAbstractions.zeros(backend, T1, size(output_estimate)...)
    ind = argmax(output_estimate, dims=1)
    # get tensor of maximum elements
    tensor_of_maximum_elements[ind] .= T1(1)
    (size(dl.output, 3)-sum(abs.(dl.output - tensor_of_maximum_elements))/T1(2))/size(dl.output, 3)
end

accuracy(nn::NeuralNetwork, dl::DataLoader) = accuracy(nn.model, nn.params, dl)

Base.eltype(::DataLoader{T}) where T = T