# MNIST tutorial 

This is a short tutorial that shows how we can use `GeometricMachineLearning` to build a vision transformer and apply it for MNIST, while also putting some of the weights on a manifold. This is also the result presented in [brantner2023generalizing](@cite).

First, we need to import the relevant packages: 

```julia
using GeometricMachineLearning, CUDA, Plots
import Zygote, MLDatasets, KernelAbstractions
```

For the AD routine we here use the `GeometricMachineLearning` default and we get the dataset from `MLDatasets`. First we need to load the data set, and put it on GPU (if you have one):

```julia
train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]
train_x = train_x |> cu 
test_x = test_x |> cu 
train_y = train_y |> cu 
test_y = test_y |> cu
```

`GeometricMachineLearning` has built-in data loaders that make it particularly easy to handle data: 

```julia
patch_length = 7
dl = DataLoader(train_x, train_y, patch_length=patch_length)
dl_test = DataLoader(train_x, train_y, patch_length=patch_length)
```

Here `patch_length` indicates the size one patch has. One image in MNIST is of dimension ``28\times28``, this means that we decompose this into 16 ``(7\times7)`` images (also see [brantner2023generalizing](@cite)).

We next define the model with which we want to train:

```julia
model = ClassificationTransformer(dl, n_heads=n_heads, n_layers=n_layers, Stiefel=true)
```

Here we have chosen a `ClassificationTransformer`, i.e. a composition of a specific number of transformer layers composed with a classification layer. We also set the *Stiefel option* to `true`, i.e. we are optimizing on the Stiefel manifold.

We now have to initialize the neural network weights. This is done with the constructor for `NeuralNetwork`:

```julia
backend = KernelAbstractions.get_backend(dl)
T = eltype(dl)
nn = NeuralNetwork(model, backend, T)
```

And with this we can finally perform the training:

```julia
# an instance of batch is needed for the optimizer
batch = Batch(batch_size)

optimizer_instance = Optimizer(AdamOptimizer(), nn)

# this prints the accuracy and is optional
println("initial test accuracy: ", accuracy(Ψᵉ, ps, dl_test), "\n")

loss_array = optimizer_instance(nn, dl, batch, n_epochs)

println("final test accuracy: ", accuracy(Ψᵉ, ps, dl_test), "\n")
```

It is instructive to play with `n_layers`, `n_epochs` and the Stiefel property.

```@bibliography
Pages = []
Canonical = false

brantner2023generalizing
```