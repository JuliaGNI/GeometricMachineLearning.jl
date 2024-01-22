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

```@example matrix
using GeometricMachineLearning # hide 

qp_data = (q = rand(Float32, 2, 10), p = rand(Float32, 2, 10))
dl = DataLoader(qp_data; autoencoder=true)

batch = Batch(3)
batch(dl)
```

```@example matrix
using GeometricMachineLearning # hide 

qp_data = (q = rand(Float32, 2, 10), p = rand(Float32, 2, 10))
dl = DataLoader(qp_data; autoencoder=false) # false is default 

batch = Batch(3)
batch(dl)
```

Not that the routines are implemented in such a way that no two indices appear double. 

## Sampling from a tensor 

We can also sample tensor data.

```@example tensor
using GeometricMachineLearning # hide 

qp_data = (q = rand(Float32, 2, 8, 5), p = rand(Float32, 2, 8, 5))
dl = DataLoader(qp_data)

# also specify sequence length here
batch = Batch(3, 5)
batch(dl)
```