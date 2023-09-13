# MNIST tutorial 

This is a short tutorial that shows how we can use `GeometricMachineLearning` to build a vision transformer and apply it for MNIST, while also putting some of the weights on a manifold. 

First, we need to import the relevant packages: 

```julia
using GeometricMachineLearning, CUDA
import Zygote, MLDatasets
```

In this example `Zygote` as an AD routine and we get the dataset from `MLDatasets`. First we need to load the data set, and put it on GPU (if you have one):

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
dl = DataLoader(train_x, train_y, batch_size=512, patch_length=patch_length)
dl_test = DataLoader(train_x, train_y, batch_size=length(y), patch_length=patch_length)
```

The second line in the above code snippet indicates that we use the entire data set as one "batch" when processing the test set. For training, the batch size was here set to 512. 

```julia
ps = initialparameters(backend, eltype(dl.data), Ψᵉ) 

optimizer_instance = Optimizer(o, ps)

println("initial test accuracy: ", accuracy(Ψᵉ, ps, dl_test), "\n")

progress_object = Progress(n_training_steps; enabled=true)

loss_array = zeros(eltype(train_x), n_training_steps)
for i in 1:n_training_steps
    redraw_batch!(dl)
    # get rid of try catch statement. This softmax issue should be solvable!
    loss_val, pb = try Zygote.pullback(ps -> loss(Ψᵉ, ps, dl), ps)
    catch
        loss_array[i] = loss_array[i-1] 
        continue 
    end
    dp = pb(one(loss_val))[1]

    optimization_step!(optimizer_instance, Ψᵉ, ps, dp)
    ProgressMeter.next!(progress_object; showvalues = [(:TrainingLoss, loss_val)])   
    loss_array[i] = loss_val
end

println("final test accuracy: ", accuracy(Ψᵉ, ps, dl_test), "\n")
```