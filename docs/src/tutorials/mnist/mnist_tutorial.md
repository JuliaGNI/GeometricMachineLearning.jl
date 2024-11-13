```@raw latex
In this chapter we discuss examples of improving transformers by imbuing them with structure\footnotemark[1]. We show two examples: an example of using the vision transformer where we put orthogonality constraints on some of the weights (which effectively leads to manifold optimization) and using the volume-preserving transformer to learn the dynamics of a rigid body. At the end we further compare two different approaches to realizing the volume-preserving transformer.

\footnotetext[1]{Technically the linear symplectic transformer from the previous chapter could also be included here, but because of the very severe modification/limitation of the attention layer we performed there, we decided against it.}
```

# MNIST Tutorial 

In this tutorial we show how we can use `GeometricMachineLearning` to build a vision transformer and apply it for MNIST [deng2012mnist](@cite), while also putting some of the weights on a manifold. This is also the result presented in [brantner2023generalizing](@cite).

We get the dataset from [`MLDatasets`](https://github.com/JuliaML/MLDatasets.jl). Before we use it we allocate it on gpu with `cu` from `CUDA.jl` [besard2018juliagpu](@cite):

```@setup mnist
using GeometricMachineLearning
import MLDatasets
train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]

nothing # hide
```

```julia
using MLDatasets
using CUDA
train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]

train_x = train_x |> cu
train_y = train_y |> cu
test_x = test_x |> cu
test_y = test_y |> cu
```

Next we call [`DataLoader`](@ref) on these data. For this we first need to specify a *patch length*[^1].

[^1]: When [`DataLoader`](@ref) is called this way it uses [`split_and_flatten`](@ref) internally.

```@eval
Main.remark(raw"In order to apply the transformer to a data set we should typically cast these data into a *time series format*. MNIST images are pictures with ``28\times28`` pixels. Here we cast these images into *time series* of length 16, so one image is represented by a matrix ``\in\mathbb{R}^{49\times{}16}``.")
```

```@example mnist
const patch_length = 7
dl = DataLoader(train_x, train_y, patch_length = patch_length; suppress_info = true)

nothing # hide
```

Here we called [`DataLoader`](@ref) on a tensor and a vector of integers (targets) as input. [`DataLoader`](@ref) automatically converts the data to the correct input format for easy handling. This is visualized below:

```@example
Main.include_graphics("mnist_visualization"; caption = "Visualization of how the data are preprocessed. An image is first split and then flattened. ", width = .8) # hide
```

Internally [`DataLoader`](@ref) calls [`split_and_flatten`](@ref) which splits each image into a number of *patches* according to the keyword arguments `patch_length` and `number_of_patches`. We also load the test data with [`DataLoader`](@ref):

```@example mnist
dl_test = DataLoader(test_x, test_y, patch_length=patch_length)

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

We still have to initialize the optimizers:

```@example mnist
const batch_size = 2048
const n_epochs = 500
# an instance of batch is needed for the optimizer
batch = Batch(batch_size, dl)

opt1 = Optimizer(AdamOptimizer(T), nn1)
opt2 = Optimizer(AdamOptimizer(T), nn2)

nothing # hide
```

And with this we can finally perform the training:

```julia
loss_array1 = opt1(nn1, dl, batch, n_epochs, FeedForwardLoss())
loss_array2 = opt2(nn2, dl, batch, n_epochs, FeedForwardLoss())
```

We furthermore optimize the second neural network (with weights on the manifold) with the [`GradientOptimizer`](@ref) and the [`MomentumOptimizer`](@ref):

```@example mnist
nn3 = NeuralNetwork(model2, backend, T)
nn4 = NeuralNetwork(model2, backend, T)

opt3 = Optimizer(GradientOptimizer(T(0.001)), nn3)
opt4 = Optimizer(MomentumOptimizer(T(0.001), T(0.5)), nn4)

nothing # hide
```


For training we use the same data, the same batch and the same number of epochs:
```julia
loss_array3 = opt3(nn3, dl, batch, n_epochs, FeedForwardLoss())
loss_array4 = opt4(nn4, dl, batch, n_epochs, FeedForwardLoss())
```

And we get the following result:

```@setup mnist
using JLD2
using CairoMakie

data = load("mnist_parameters.jld2")
loss_array1 = data["loss_array1"]
loss_array2 = data["loss_array2"]
loss_array3 = data["loss_array3"]
loss_array4 = data["loss_array4"]

accuracy_score1 = data["accuracy_score1"]
accuracy_score2 = data["accuracy_score2"]
accuracy_score3 = data["accuracy_score3"]
accuracy_score4 = data["accuracy_score4"]

_nnp(ps::Tuple) = NeuralNetworkParameters{Tuple(Symbol("L$(i)") for i in 1:length(ps))}(ps)
nn1 = NeuralNetwork(nn1.architecture, nn1.model, _nnp(data["nn1weights"]), CPU())
nn2 = NeuralNetwork(nn2.architecture, nn2.model, _nnp(data["nn2weights"]), CPU())
nn3 = NeuralNetwork(nn3.architecture, nn3.model, _nnp(data["nn3weights"]), CPU())
nn4 = NeuralNetwork(nn4.architecture, nn4.model, _nnp(data["nn4weights"]), CPU())

morange = RGBf(255 / 256, 127 / 256, 14 / 256) # hide
mred = RGBf(214 / 256, 39 / 256, 40 / 256) # hide
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256) # hide
mblue = RGBf(31 / 256, 119 / 256, 180 / 256) # hide

function make_error_plot(; theme = :dark) # hide
textcolor = theme == :dark ? :white : :black # hide
fig = Figure(; backgroundcolor = :transparent)
ax = Axis(fig[1, 1]; 
    backgroundcolor = :transparent,
    bottomspinecolor = textcolor, 
    topspinecolor = textcolor,
    leftspinecolor = textcolor,
    rightspinecolor = textcolor,
    xtickcolor = textcolor, 
    ytickcolor = textcolor,
    xticklabelcolor = textcolor,
    yticklabelcolor = textcolor,
    xlabel="Epoch", 
    ylabel="Training loss",
    xlabelcolor = textcolor,
    ylabelcolor = textcolor,
    )

lines!(ax, loss_array1, label="Adam", color=mblue)
lines!(ax, loss_array2, label="Adam + Stiefel", color=mred)
lines!(ax, loss_array3, label="Gradient + Stiefel", color=mpurple)
lines!(ax, loss_array4, label="Momentum + Stiefel", color=morange)
axislegend(; position = (.82, .75), backgroundcolor = :transparent, labelcolor = textcolor) # hide
fig_name = theme == :dark ? "mnist_training_loss_dark.png" : "mnist_training_loss.png" # hide
save(fig_name, fig; px_per_unit = 1.2) # hide
end # hide
make_error_plot(; theme = :dark) # hide
make_error_plot(; theme = :light) # hide
```

```@example
Main.include_graphics("mnist_training_loss"; width = .7, caption = raw"Comparison between the standard Adam optimizer (blue), the Adam optimizer with weights on the Stiefel manifold (purple), the gradient optimizer with weights on the Stiefel manifold (purple) and the momentum optimizer with weights on the Stiefel manifold (orange). ") # hide
```

```@eval
Main.remark(raw"We see that the loss value for the Adam optimizer without parameters on the Stiefel manifold is stuck at around 1.34 which means that it *always predicts the same value*. So in 1 out of ten cases we have error 0 and in 9 out of ten cases we have error ``\sqrt{2}``, giving
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \sqrt{2\frac{9}{10}} = 1.342,
" * Main.indentation * raw"```
" * Main.indentation * raw"which is what we see in the error plot.")
```

We can also call [`GeometricMachineLearning.accuracy`](@ref) to obtain the test accuracy instead of the training error:

```@example mnist
using GeometricMachineLearning: accuracy # hide
(accuracy(nn1, dl_test), accuracy(nn2, dl_test), accuracy(nn3, dl_test), accuracy(nn4, dl_test))
```

```@eval
Main.remark(raw"We note here that conventional convolutional neural networks and other vision transformers achieve much better accuracy on MNIST in a training time that is often shorter than what we presented here. Our aim here is not to outperform existing neural networks in terms of accuracy on image classification problems, but to demonstrate two things: (i) in many cases putting weights on the Stiefel manifold (which is a compact space) can enable training that would otherwise not be possible and (ii) as is the case with standard Adam, the manifold version also seems to achieve similar performance gain over the gradient and momentum optimizer. Both of these observations are demonstrated figure above.")
```

## Library Functions

```@docs
split_and_flatten
GeometricMachineLearning.accuracy
GeometricMachineLearning.onehotbatch
GeometricMachineLearning.ClassificationLayer
```

```@raw latex
\begin{comment}
```

## References

```@bibliography
Pages = []
Canonical = false

brantner2023generalizing
```

```@raw latex
\end{comment}
```