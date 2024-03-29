# Data Loader 

```@eval
using GeometricMachineLearning, Markdown
Markdown.parse(description(Val(:DataLoader)))
```

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

```@eval
using GeometricMachineLearning, Markdown 
Markdown.parse(description(Val(:data_loader_constructor_matrix)))
```

```@example 
using GeometricMachineLearning # hide

SnapshotMatrix = rand(Float32, 10, 100)

dl = DataLoader(SnapshotMatrix; autoencoder=false)
dl.input_time_steps
```


```@eval
using GeometricMachineLearning, Markdown
Markdown.parse(description(Val(:data_loader_for_named_tuple)))
```

```@example
using GeometricMachineLearning # hide

SymplecticSnapshotTensor = (q = rand(Float32, 10, 100, 5), p = rand(Float32, 10, 100, 5))

dl = DataLoader(SymplecticSnapshotTensor)
```

## Convenience functions 

```@eval
using GeometricMachineLearning, Markdown
Markdown.parse(description(Val(:Batch)))
```

```@eval
using GeometricMachineLearning, Markdown
Markdown.parse(description(Val(:batch_functor_matrix)))
```

```@example 
using GeometricMachineLearning # hide

matrix_data = rand(Float32, 2, 10)
dl = DataLoader(matrix_data)

batch = Batch(3)
batch(dl)
```

This also works if the data are in ``qp`` form: 

```@example
using GeometricMachineLearning # hide 

qp_data = (q = rand(Float32, 2, 10), p = rand(Float32, 2, 10))
dl = DataLoader(qp_data; autoencoder=true)

batch = Batch(3)
batch(dl)
```

```@example
using GeometricMachineLearning # hide 

qp_data = (q = rand(Float32, 2, 10), p = rand(Float32, 2, 10))
dl = DataLoader(qp_data; autoencoder=false) # false is default 

batch = Batch(3)
batch(dl)
```

Specifically the routines do the following: 
1. ``\mathtt{n\_indices}\leftarrow \mathtt{n\_params}\lor\mathtt{input\_time\_steps}`` 
2. ``\mathtt{indices} \leftarrow \mathtt{shuffle}(\mathtt{1:\mathtt{n\_indices}})``
3. ``\mathcal{I}_i \leftarrow \mathtt{indices[(i - 1)} \cdot \mathtt{batch\_size} + 1 \mathtt{:} i \cdot \mathtt{batch\_size]}\text{ for }i=1, \ldots, (\mathrm{last} -1)``
4. ``\mathcal{I}_\mathtt{last} \leftarrow \mathtt{indices[}(\mathtt{n\_batches} - 1) \cdot \mathtt{batch\_size} + 1\mathtt{:end]}``

Note that the routines are implemented in such a way that no two indices appear double. 

## Sampling from a tensor 

We can also sample tensor data.

```@example
using GeometricMachineLearning # hide 

qp_data = (q = rand(Float32, 2, 8, 3), p = rand(Float32, 2, 8, 3))
dl = DataLoader(qp_data)

# also specify sequence length here
batch = Batch(4, 5)
batch(dl)
```

Sampling from a tensor is done the following way (``\mathcal{I}_i`` again denotes the batch indices for the ``i``-th batch): 
1. ``\mathtt{time\_indices} \leftarrow \mathtt{shuffle}(\mathtt{1:}(\mathtt{input\_time\_steps} - \mathtt{seq\_length})``
2. ``\mathtt{parameter\_indices} \leftarrow \mathtt{shuffle}(\mathtt{1:n\_params})``
3. ``\mathtt{complete\_indices} \leftarrow \mathtt{Iterators.product}(\mathtt{time\_indices}, \mathtt{parameter\_indices}) \mathtt{|> collect |> vec}``
3. ``\mathcal{I}_i \leftarrow \mathtt{complete\_indices[}(i - 1) \cdot \mathtt{batch\_size} + 1 : i \cdot \mathtt{batch\_size]}\text{ for }i=1, \ldots, (\mathrm{last} -1)``
4. ``\mathcal{I}_\mathrm{last} \leftarrow \mathtt{complete\_indices[}(\mathrm{last} - 1) \cdot \mathtt{batch\_size} + 1\mathtt{:end]}``

This algorithm can be visualized the following way (here `batch_size = 4`):

```@example 
import Images, Plots # hide
if Main.output_type == :html # hide
    HTML("""<object type="image/svg+xml" class="display-light-only" data=$(joinpath(Main.buildpath, "../tikz/tensor_sampling.png"))></object>""") # hide
else # hide
    Plots.plot(Images.load("../tikz/tensor_sampling.png"), axis=([], false)) # hide
end # hide
```

```@example
if Main.output_type == :html # hide
    HTML("""<object type="image/svg+xml" class="display-dark-only" data=$(joinpath(Main.buildpath, "../tikz/tensor_sampling_dark.png"))></object>""") # hide
end # hide
```

Here the sampling is performed over the second axis (the *time step dimension*) and the third axis (the *parameter dimension*). Whereas each block has thickness 1 in the ``x`` direction (i.e. pertains to a single parameter), its length in the ``y`` direction is `seq_length`. In total we sample as many such blocks as the batch size is big. By construction those blocks are never the same throughout a training epoch but may intersect each other!