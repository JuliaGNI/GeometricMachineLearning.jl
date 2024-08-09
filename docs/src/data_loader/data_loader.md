# The Data Loader 

The `DataLoader` in `GeometricMachineLearning` is designed to make training convenient. 

The data loader can be called with various types of arrays as input, for example a [snapshot matrix](snapshot_matrix.md):

```@example
using GeometricMachineLearning # hide

SnapshotMatrix = rand(Float32, 10, 100)

dl = DataLoader(SnapshotMatrix)
```

or a snapshot tensor: 

```@example
using GeometricMachineLearning # hide

SnapshotTensor = rand(Float32, 10, 100, 5)

dl = DataLoader(SnapshotTensor)
```

Here the `DataLoader` has different properties `:RegularData` and `:TimeSeries`. This indicates that in the first case we treat all columns in the input tensor independently (this is mostly used for autoencoder problems), whereas in the second case we have *time series-like data*, which are mostly used for integration problems. The default when using a matrix is `:RegularData` and the default when using a tensor is `:TimeSeries`.

We can also treat a problem with a matrix as input as a time series-like problem by providing an additional keyword argument: `autoencoder=false`:

```@example 
using GeometricMachineLearning # hide

SnapshotMatrix = rand(Float32, 10, 100)

dl = DataLoader(SnapshotMatrix; autoencoder=false)
dl.input_time_steps
```

If we deal with hamiltonian systems we typically split the coordinates into a ``q`` and a ``p`` part. Such data can also be used as input arguments for `DataLoader`:

```@example named_tuple_tensor
using GeometricMachineLearning # hide

SymplecticSnapshotTensor = (q = rand(Float32, 10, 100, 5), p = rand(Float32, 10, 100, 5))

dl = DataLoader(SymplecticSnapshotTensor)
```

The dimension of the system is then the sum of the dimensions of the ``q`` and the ``p`` component:

```@example named_tuple_tensor
dl.input_dim
```

## The `Batch` struct  

If we want to draw mini batches from a data loader, we need to allocate an instance of [`Batch`](@ref). If we call the corresponding functor on an instance of [`DataLoader`](@ref) we get the following result:

```@example 
using GeometricMachineLearning # hide

matrix_data = rand(Float32, 2, 10)
dl = DataLoader(matrix_data; autoencoder = true)

batch = Batch(3)
batch(dl)
```

The output of applying the batch functor is always of the form: 

```math
([(b_{1,1}^t, b_{1,1}^p), (b_{1,2}^t, b_{1,2}^p), \ldots], [(b_{2,1}^t, b_{2, 1}^p), (b_{2, 2}^t, b_{2, 2}^p), \ldots], [(b_{3, 1}^t, b_{3, 2}^p), \ldots], \ldots),
```

so it is a tuple of vectors of tuples. One vector represents one batch. the tuples in one vector always have two entries: a *time index* and a *parameter index,* indicated by the superscripts ``t`` and ``p`` respectively in the equation above.

This also works if the data are in ``(q, p)`` form:

```@example
using GeometricMachineLearning # hide

qp_data = (q = rand(Float32, 2, 10), p = rand(Float32, 2, 10))
dl = DataLoader(qp_data; autoencoder = true)

batch = Batch(3)
batch(dl)
```

In those two examples the `autoencoder` keyword was set to `true` (the default). This is why the first index was always `1`. This changes if we set `autoencoder = false`: 

```@example
using GeometricMachineLearning # hide

qp_data = (q = rand(Float32, 2, 10), p = rand(Float32, 2, 10))
dl = DataLoader(qp_data; autoencoder = false) # false is default 

batch = Batch(3)
batch(dl)
```

Specifically the routines do the following: 
1. ``\mathtt{n\_indices}\leftarrow \mathtt{n\_params}\lor\mathtt{input\_time\_steps},`` 
2. ``\mathtt{indices} \leftarrow \mathtt{shuffle}(\mathtt{1:\mathtt{n\_indices}}),``
3. ``\mathcal{I}_i \leftarrow \mathtt{indices[(i - 1)} \cdot \mathtt{batch\_size} + 1 \mathtt{:} i \cdot \mathtt{batch\_size]}\text{ for }i=1, \ldots, (\mathrm{last} -1),``
4. ``\mathcal{I}_\mathtt{last} \leftarrow \mathtt{indices[}(\mathtt{n\_batches} - 1) \cdot \mathtt{batch\_size} + 1\mathtt{:end]}.``

Note that the routines are implemented in such a way that no two indices appear double, i.e. we *sample without replacement*. 

## Sampling from a tensor 

We can also sample tensor data.

```@example
using GeometricMachineLearning # hide

qp_data = (q = rand(Float32, 2, 20, 3), p = rand(Float32, 2, 20, 3))
dl = DataLoader(qp_data)

# also specify sequence length here
batch = Batch(4, 5)
batch(dl)
```

Sampling from a tensor is done the following way (``\mathcal{I}_i`` again denotes the batch indices for the ``i``-th batch): 
1. ``\mathtt{time\_indices} \leftarrow \mathtt{shuffle}(\mathtt{1:}(\mathtt{input\_time\_steps} - \mathtt{seq\_length} - \mathtt{prediction_window}),``
2. ``\mathtt{parameter\_indices} \leftarrow \mathtt{shuffle}(\mathtt{1:n\_params}),``
3. ``\mathtt{complete\_indices} \leftarrow \mathtt{product}(\mathtt{time\_indices}, \mathtt{parameter\_indices}),``
3. ``\mathcal{I}_i \leftarrow \mathtt{complete\_indices[}(i - 1) \cdot \mathtt{batch\_size} + 1 : i \cdot \mathtt{batch\_size]}\text{ for }i=1, \ldots, (\mathrm{last} -1),``
4. ``\mathcal{I}_\mathrm{last} \leftarrow \mathtt{complete\_indices[}(\mathrm{last} - 1) \cdot \mathtt{batch\_size} + 1\mathtt{:end]}.``

This algorithm can be visualized the following way (here `batch_size = 4`):

```@example
Main.include_graphics("../tikz/tensor_sampling") # hide
```

Here the sampling is performed over the second axis (the *time step dimension*) and the third axis (the *parameter dimension*). Whereas each block has thickness 1 in the ``x`` direction (i.e. pertains to a single parameter), its length in the ``y`` direction is `seq_length`. In total we sample as many such blocks as the batch size is big. By construction those blocks are never the same throughout a training epoch but may intersect each other!

## Library Functions

```@docs
DataLoader
Batch
GeometricMachineLearning.number_of_batches
GeometricMachineLearning.convert_input_and_batch_indices_to_array
```