# Projection and Reduction Errors of Reduced Models

Two errors that are of very big importance in reduced order modeling are the **projection** and the **reduction error**. During training one typically aims to miminimze the projection error, but for the actual application of the model the reduction error is often more important. 

## Projection Error 

The projection error computes how well a reduced basis, represented by the reduction $\mathcal{P}$ and the reconstruction $\mathcal{R}$, can represent the data with which it is build. In mathematical terms: 

```math
e_\mathrm{proj}(\mu) := 
    \frac{|| \mathcal{R}\circ\mathcal{P}(M) - M ||}{|| M ||},
```
where $||\cdot||$ is the Frobenius norm (one could also optimize for different norms).

## Reduction Error

The reduction error measures how far the reduced system diverges from the full-order system during integration (online stage). In mathematical terms (and for a single initial condition): 

```math
e_\mathrm{red}(\mu) := \sqrt{
    \frac{\sum_{t=0}^K|| \mathbf{x}^{(t)}(\mu) - \mathcal{R}(\mathbf{x}^{(t)}_r(\mu)) ||^2}{\sum_{t=0}^K|| \mathbf{x}^{(t)}(\mu) ||^2}
},
```
where $\mathbf{x}^{(t)}$ is the solution of the FOM at point $t$ and $\mathbf{x}^{(t)}_r$ is the solution of the ROM (in the reduced basis) at point $t$. The reduction error, as opposed to the projection error, not only measures how well the solution manifold is represented by the reduced basis, but also measures how well the FOM dynamics are approximated by the ROM dynamics (via the induced vector field on the reduced basis).