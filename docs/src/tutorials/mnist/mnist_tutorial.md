# MNIST Tutorial 

This is a short tutorial that shows how we can use `GeometricMachineLearning` to build a vision transformer and apply it for MNIST [deng2012mnist](@cite), while also putting some of the weights on a manifold. This is also the result presented in [brantner2023generalizing](@cite).

First, we need to import the relevant packages: 

```@example mnist
using GeometricMachineLearning
import MLDatasets
```

For the AD routine we here use the `GeometricMachineLearning` default and we get the dataset from [`MLDatasets`](https://github.com/JuliaML/MLDatasets.jl). First we need to load the data set and preprocess it:

```@example mnist
train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]

const patch_length = 7
dl = DataLoader(train_x, train_y, patch_length = patch_length)

nothing # hide
```

Here we called [`DataLoader`](@ref) on a tensor and a vector of integers (targets) as input. [`DataLoader`](@ref) automatically converts the data to the correct input format for easy handling. This is visualized below:

```@example
Main.include_graphics("mnist_visualization"; caption = "Visualization of how the data are preprocessed.") # hide
```

Internally [`DataLoader`](@ref) calls [`split_and_flatten`](@ref) which splits each image into a number of *patches* according to the keyword arguments `patch_length` and `number_of_patches`. We also load the test data with [`DataLoader`](@ref):

```@example mnist
dl_test = DataLoader(train_x, train_y, patch_length=patch_length)

nothing # hide
```

We next define the model with which we want to train:

```@example mnist
const n_heads = 7
const L = 16
const add_connection = false

model1 = ClassificationTransformer(dl; 
                                    n_heads = n_heads, 
                                    L = L, 
                                    add_connection = add_connection, 
                                    Stiefel = false)
model2 = ClassificationTransformer(dl; 
                                    n_heads = n_heads, 
                                    L = L, 
                                    add_connection = add_connection, 
                                    Stiefel = true)

nothing # hide
```

Here we have chosen a [`ClassificationTransformer`](@ref), i.e. a composition of a specific number of transformer layers composed with a classification layer. We also set the *Stiefel option* to `true`, i.e. we are optimizing on the Stiefel manifold.

We now have to initialize the neural network weights. This is done with the constructor for `NeuralNetwork`:

```@example mnist
backend = GeometricMachineLearning.get_backend(dl)
T = eltype(dl)
nn1 = NeuralNetwork(model1, backend, T)
nn2 = NeuralNetwork(model2, backend, T)

nothing # hide
```

And with this we can finally perform the training:

```julia
const batch_size = 2048
const n_epochs = 500
# an instance of batch is needed for the optimizer
batch = Batch(batch_size, dl)

opt1 = Optimizer(AdamOptimizer(T), nn1)
opt2 = Optimizer(AdamOptimizer(T), nn2)

loss_array1 = opt1(nn1, dl, batch, n_epochs, GeometricMachineLearning.ClassificationTransformerLoss())
loss_array2 = opt2(nn2, dl, batch, n_epochs, GeometricMachineLearning.ClassificationTransformerLoss())
```

We furthermore optimize the second model (with weights on the manifold) with the [`GradientOptimizer`](@ref) and the [`MomentumOptimizer`](@ref):

```julia
nn3 = NeuralNetwork(model2, backend, T)
nn4 = NeuralNetwork(model2, backend, T)

opt3 = Optimizer(GradientOptimizer(T(0.001)), nn3)
opt4 = Optimizer(MomentumOptimizer(T(0.001), T(0.5)), nn4)

loss_array3 = opt3(nn3, dl, batch, n_epochs, GeometricMachineLearning.ClassificationTransformerLoss())
loss_array4 = opt4(nn4, dl, batch, n_epochs, GeometricMachineLearning.ClassificationTransformerLoss())
```


## Library Functions

```@docs
split_and_flatten
GeometricMachineLearning.accuracy
GeometricMachineLearning.onehotbatch
GeometricMachineLearning.ClassificationLayer
```

## References

```@bibliography
Pages = []
Canonical = false

brantner2023generalizing
```