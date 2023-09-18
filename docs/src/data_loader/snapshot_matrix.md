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