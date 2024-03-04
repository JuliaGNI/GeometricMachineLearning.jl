# Snapshot matrix

The snapshot matrix stores solutions of the high-dimensional ODE (obtained from discretizing a PDE). This is then used to construct [reduced bases](../reduced_order_modeling/autoencoder.md) in a data-driven way. So (for a single parameter[^1]) the snapshot matrix takes the following form: 

[^1]: If we deal with a parametrized PDE then there are **two stages** at which the snapshot matrix has to be processed: the offline stage and the online stage. 

```math
M = \left[\begin{array}{c:c:c:c}
\hat{u}_1(t_0) &  \hat{u}_1(t_1) & \quad\ldots\quad & \hat{u}_1(t_f) \\
\hat{u}_2(t_0) &  \hat{u}_2(t_1) & \ldots & \hat{u}_2(t_f) \\
\hat{u}_3(t_0) &  \hat{u}_3(t_1) & \ldots & \hat{u}_3(t_f) \\
\ldots &  \ldots & \ldots & \ldots \\
\hat{u}_{2N}(t_0) &  \hat{u}_{2N}(t_1) & \ldots & \hat{u}_{2N}(t_f) \\
\end{array}\right].
```

In the above example we store a matrix whose first axis is the system dimension (i.e. a state is an element of ``\mathbb{R}^{2n}``) and the second dimension gives the time step. 

The starting point for using the snapshot matrix as data for a machine learning model is that all the columns of ``M`` live on a lower-dimensional [solution manifold](../reduced_order_modeling/autoencoder.md) and we can use techniques such as *POD* and *autoencoders* to find this solution manifold. We also note that the second axis of ``M`` does not necessarily indicate time but can also represent various parameters (including initial conditions). The second axis in the `DataLoader` struct is therefore saved in the field `n_params`.



# Snapshot tensor 

The snapshot tensor fulfills the same role as the snapshot matrix but has a third axis that describes different initial parameters (such as different initial conditions). 

```@example 
import Images, Plots # hide
if Main.output_type == :html_output # hide
    HTML("""<object type="image/svg+xml" class="display-light-only" data=$(joinpath(Main.buildpath, "../tikz/tensor.png"))></object>""") # hide
else # hide
    Plots.plot(Images.load("../tikz/tensor.png"), axis=([], false)) # hide
end # hide
```

```@example
if Main.output_type == :html_output # hide
    HTML("""<object type="image/svg+xml" class="display-dark-only" data=$(joinpath(Main.buildpath, "../tikz/tensor_dark.png"))></object>""") # hide
end # hide
```

When drawing training samples from the snapshot tensor we also need to specify a *sequence length* (as an argument to the [`Batch`](@ref) struct). When sampling a batch from the snapshot tensor we sample over the starting point of the time interval (which is of length `seq_length`) and the third axis of the tensor (the parameters). The total number of batches in this case is ``\lceil\mathtt{(dl.input\_time_steps - batch.seq\_length) * dl.n\_params / batch.batch_size}\rceil``. 