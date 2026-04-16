# The Data Loader 

The `DataLoader` in `GeometricMachineLearning` is designed to make training convenient. 

The data loader can be called with various types of arrays as input, for example a [snapshot matrix](@ref "Snapshot Matrix"):

```@example snapshot_matrix
using GeometricMachineLearning # hide
SnapshotMatrix = [ 1; 2;; 3; 4;; 5; 6;; 7; 8;; 9; 10 ]

dl = DataLoader(SnapshotMatrix; suppress_info = true)
```

or a snapshot tensor: 

```@example snapshot_tensor
using GeometricMachineLearning # hide
SnapshotTensor = [ 1;  2;; 3; 4;; 5; 6;; 7; 8;; 9; 10 ;;;]

dl = DataLoader(SnapshotTensor; suppress_info = true)
```

Here the `DataLoader` has different properties `:RegularData` and `:TimeSeries`. This indicates that in the first case we treat all columns in the input tensor independently; this is mostly used for [autoencoder problems](@ref "Autoencoders"). In the second case we have *time series-like data*, which are mostly used for [integration problems](@ref "Neural Network Integrators"). As shown above the default when using a matrix is `:RegularData` and the default when using a tensor is `:TimeSeries`.

We can also treat a problem with a matrix as input as a time series-like problem by providing an additional keyword argument: `autoencoder=false`:

```@example snapshot_matrix
dl = DataLoader(SnapshotMatrix; autoencoder=false, suppress_info = true)
@assert typeof(dl) == DataLoader{Int64, Array{Int64, 3}, Nothing, :TimeSeries} # hide
dl |> typeof
```

If we deal with hamiltonian systems we typically split the coordinates into a ``q`` and a ``p`` part. Such data can also be used as input arguments for `DataLoader`:

```@example snapshot_tensor
using GeometricMachineLearning # hide
SymplecticSnapshotTensor = (q = SnapshotTensor, p = SnapshotTensor)
dl = DataLoader(SymplecticSnapshotTensor)
@assert typeof(dl) == DataLoader{Int64, @NamedTuple{q::Array{Int64, 3}, p::Array{Int64, 3}}, Nothing, :TimeSeries} # hide
dl |> typeof
```

The dimension of the system is then the sum of the dimensions of the ``q`` and the ``p`` component:

```@example snapshot_tensor
@assert dl.input_dim == 4 # hide
dl.input_dim
```

## Drawing Batches with `GeometricMachineLearning` 

If we want to draw mini batches from a data set, we need to allocate an instance of [`Batch`](@ref). If we call the corresponding functor on an instance of [`DataLoader`](@ref) we get the following result[^1]:

[^1]: We first demonstrate how to sample data on the example of a matrix. The case of sampling from a tensor is slightly more complicated and is explained below.

```@example batches
using GeometricMachineLearning # hide
import Random # hide
Random.seed!(123) # hide
matrix_data = [ 1 2 3 4  5;
                6 7 8 9 10]
dl = DataLoader(matrix_data; autoencoder = true)

batch = Batch(3)
batches = batch(dl)
```

The output of applying the batch functor is always of the form: 

```math
([(b_{1,1}^t, b_{1,1}^p), (b_{1,2}^t, b_{1,2}^p), \ldots], [(b_{2,1}^t, b_{2, 1}^p), (b_{2, 2}^t, b_{2, 2}^p), \ldots], [(b_{3, 1}^t, b_{3, 2}^p), \ldots], \ldots),
```

so it is a tuple of vectors of tuples. One vector represents one batch:

```@example batches
for (minibatch, i) in zip(batches[1], axes(batches[1], 1))
    println(stdout, minibatch[1], " = bᵗ₁" * join('₀' + d for d in digits(i)))
    println(stdout, minibatch[2], " = bᵖ₁" * join('₀' + d for d in digits(i)))
    println()
end
nothing # hide
```

The tuples that make up this vector always have two entries: a *time index* ``b^t_{1i}`` and a *parameter index* ``b^p_{1i}`` indicated by the superscripts ``t`` and ``p`` respectively. Because `dl` in this example is of `autoencoder` type, the time index is always one. The parameter index differs. Because the input to `DataLoader` was a ``2\times5`` matrix and we specified a batch size of three, there are two batches in total. The second batch is:

```@example batches
@assert length(batches) == 2 # hide
@assert length(batches[1]) + length(batches[2]) == 5 # hide
for (minibatch, i) in zip(batches[2], axes(batches[2], 1))
    println(stdout, minibatch[1], " = bᵗ₁" * join('₀' + d for d in digits(i)))
    println(stdout, minibatch[2], " = bᵖ₁" * join('₀' + d for d in digits(i)))
    println()
end
nothing # hide
```

Looking at the first and the second batch together, we see that we sample with replacement, i.e. all indices ``b^p_{1i} = 1, 2, 3, 4, 5`` appear. This also works if the data are in ``(q, p)`` form:

```@example
using GeometricMachineLearning # hide
qp_data = (q = rand(Float32, 2, 5), p = rand(Float32, 2, 5))
dl = DataLoader(qp_data; autoencoder = true)

batch = Batch(3)
batch(dl)
```

In those two examples the `autoencoder` keyword was set to `true` (the default). This is why the first index was always `1`. This changes if we set `autoencoder = false`: 

```@example
using GeometricMachineLearning # hide
qp_data = (q = rand(Float32, 2, 5), p = rand(Float32, 2, 5))
dl = DataLoader(qp_data; autoencoder = false) # false is default 

batch = Batch(3)
batch(dl)
```

Specifically the sampling routines do the following: 
1. ``\mathtt{n\_indices}\leftarrow \mathtt{n\_params}\lor\mathtt{input\_time\_steps},`` 
2. ``\mathtt{indices} \leftarrow \mathtt{shuffle}(\mathtt{1:\mathtt{n\_indices}}),``
3. ``\mathcal{I}_i \leftarrow \mathtt{indices[(i - 1)} \cdot \mathtt{batch\_size} + 1 \mathtt{:} i \cdot \mathtt{batch\_size]}\text{ for }i=1, \ldots, (\mathrm{last} -1),``
4. ``\mathcal{I}_\mathtt{last} \leftarrow \mathtt{indices[}(\mathtt{n\_batches} - 1) \cdot \mathtt{batch\_size} + 1\mathtt{:end]}.``

Note that the routines are implemented in such a way that no two indices appear double, i.e. we *sample without replacement*. 

## Sampling from a tensor 

We can also sample from a tensor:

```@example
using GeometricMachineLearning # hide
qp_data = (q = rand(Float32, 2, 5, 3), p = rand(Float32, 2, 5, 3))
dl = DataLoader(qp_data)

# also specify sequence length and a prediction window here
batch = Batch(4, 2, 0)
batch(dl)
```

Sampling from a tensor is done the following way (``\mathcal{I}_i`` again denotes the batch indices for the ``i``-th batch): 
1. ``\mathtt{time\_indices} \leftarrow \mathtt{shuffle}(\mathtt{1:}(\mathtt{input\_time\_steps} - \mathtt{seq\_length} - \mathtt{prediction\_window}),``
2. ``\mathtt{parameter\_indices} \leftarrow \mathtt{shuffle}(\mathtt{1:n\_params}),``
3. ``\mathtt{complete\_indices} \leftarrow \mathtt{product}(\mathtt{time\_indices}, \mathtt{parameter\_indices}),``
3. ``\mathcal{I}_i \leftarrow \mathtt{complete\_indices[}(i - 1) \cdot \mathtt{batch\_size} + 1 : i \cdot \mathtt{batch\_size]}\text{ for }i=1, \ldots, (\mathrm{last} -1),``
4. ``\mathcal{I}_\mathrm{last} \leftarrow \mathtt{complete\_indices[}(\mathrm{last} - 1) \cdot \mathtt{batch\_size} + 1\mathtt{:end]}.``

Note that we supplied two additional arguments to the [`Batch`](@ref) constructor here: `seq_length` and `prediction_window`. These two arguments specify how many time are considered in one mini batch and how long the prediction runs into the future respectively. These two numbers are explained when we talk about [structure on product spaces](@ref "How is Structure Preserved?").

This algorithm can be visualized the following way (here `batch_size = 4`):

![Visualization of sampling from a tensor. Here the batch size was specified as four, i.e. we sample four blocks.](../tikz/tensor_sampling_light.png)
![Visualization of sampling from a tensor. Here the batch size was specified as four, i.e. we sample four blocks.](../tikz/tensor_sampling_dark.png)

Here the sampling is performed over the second axis (the *time step dimension*) and the third axis (the *parameter dimension*). Whereas each block has thickness 1 in the ``x`` direction (i.e. pertains to a single parameter), its length in the ``y`` direction is `seq_length`. In total we sample as many such blocks as the batch size is big. By construction those blocks are never the same throughout a training epoch but may intersect each other!

## Library Functions

```@docs
DataLoader
Batch
Batch(::Int)
Batch(::Int, ::Int, ::Int)
GeometricMachineLearning.number_of_batches
GeometricMachineLearning.convert_input_and_batch_indices_to_array
```