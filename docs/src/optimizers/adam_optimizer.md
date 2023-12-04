# The Adam Optimizer 

The Adam Optimizer is one of the most widely (if not the most widely used) neural network optimizer. Like most modern neural network optimizers it contains a `cache` that is updated based on first-order gradient information and then, in a second step, the `cache` is used to compute a velocity estimate for updating the neural networ weights. 

Here we first describe the Adam algorithm for the case where all the weights are on a vector space and then show how to generalize this to the case where the weights are on a manifold. 

## All weights on a vector space

The cache of the Adam optimizer consists of **first and second moments**. The **first moments** $B_1$ store linear information about the current and previous gradients, and the **second moments** $B_2$ store quadratic information about current and previous gradients (all computed from a first-order gradient). 

If all the weights are on a vector space, then we directly compute updates for $B_1$ and $B_2$:
1. $B_1 \gets ((\rho_1 - \rho_1^t)/(1 - \rho_1^t))\cdot{}B_1 + (1 - \rho_1)/(1 - \rho_1^t)\cdot{}\nabla{}L,$
2. $B_2 \gets ((\rho_2 - \rho_1^t)/(1 - \rho_2^t))\cdot{}B_2 + (1 - \rho_2)/(1 - \rho_2^t)\cdot\nabla{}L\odot\nabla{}L,$

    where $\odot:\mathbb{R}^n\times\mathbb{R}^n\to\mathbb{R}^n$ is the **Hadamard product**: $[a\odot{}b]_i = a_ib_i$. $\rho_1$ and $\rho_2$ are hyperparameters. Their defaults, $\rho_1=0.9$ and $\rho_2=0.99$, are taken from (Goodfellow et al., 2016, page 301). After having updated the `cache` (i.e. $B_1$ and $B_2$) we compute a **velocity** (step 3) with which the parameters $Y_t$ are then updated (step 4).

3. $W_t\gets -\eta{}B_1/\sqrt{B_2 + \delta},$
4. $Y_{t+1} \gets Y_t + W_t,$

Here $\eta$ (with default 0.01) is the **learning rate** and $\delta$ (with default $3\cdot10^{-7}$) is a small constant that is added for stability. The division, square root and addition in step 3 are performed element-wise. 

![](../tikz/adam_optimizer.png)

## Weights on manifolds 

The problem with generalizing Adam to manifolds is that the Hadamard product $\odot$ as well as the other element-wise operations ($/$, $\sqrt{}$ and $+$ in step 3 above) lack a clear geometric interpretation. In `GeometricMachineLearning` we get around this issue by utilizing a so-called [global tangent space representation](../arrays/stiefel_lie_alg_horizontal.md).  


## References

- Goodfellow I, Bengio Y, Courville A. Deep learning[M]. MIT press, 2016.