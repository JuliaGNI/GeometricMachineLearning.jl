# The Linear Wave Equation

The **linear wave equation** is the prototypical example for a Hamiltonian PDE. It is given by (see (Buchfink et al, 2023) and (Peng and Mohseni, 2016)): 
```math
\mathcal{H}(q, p; \mu) := \frac{1}{2}\int_\Omega\mu^2(\partial_\xi{}q(t,\xi;\mu))^2 + p(t,\xi;\mu)^2d\xi,
```
with $\xi\in\Omega:=(-1/2,1/2)$ and $\mu\in\mathbb{P}:=[5/12,5/6]$. 

The PDE for to this Hamiltonian can be obtained similarly as in the ODE case:

```math
\partial_t{}q(t,\xi;\mu) = \frac{\delta{}\mathcal{H}}{\delta{}p} = p(t,\xi;\mu), \quad \partial_t{}p(t,\xi;\mu) = -\frac{\delta{}\mathcal{H}}{\delta{}q} = \mu^2\partial_{\xi{}\xi}q(t,\xi;\mu).
```
In order to now obtain a Hamiltonian ODE from this Hamiltonian PDE, we have to discretize $\mathcal{H}$ directly, for example with: 
```math
\mathcal{H}_h(z) = \sum_{i=1}^n\frac{\Delta{}x}{2}\right[p_i^2 + \mu^2\frac{(q_i - q_{i-1})^2 + (q_{i+1} - q_i)^2}{2\Delta{}x}\left] = z^TKz,
```
where the matrix $K$ contains elements of the form: 
```math
k_{ij} = \begin{cases}  \frac{\mu^2}{4\Delta{}x} &\text{if (i,j)\in\{(0,0),(N+1,N+1)\} }, \\
                        -\frac{\mu^2}{2\Delta{}x} & \text{if $(i,j)=(1,0)$ or $(i,j)=(N,N+1)$} \\
                        \frac{3\m^2}{4\Delta{}x} & \text{if $(i,j)\in\{(1,1),(N,N)\}$} \\
                        \frac{\mu^2}{\Delta{}x} & \text{if $i=j$ and $i\in\{2,\ldots,(N-2)\}$} \\ 
                        -\frac{\mu^2}{2\Delta{}x} & \text{if $|i-j|=1$ and $i,j\nin\{0,n+1\}$} \\
                        0 & \text{else}
                          \end{cases}
```


## References 
- Buchfink, Patrick, Silke Glas, and Bernard Haasdonk. "Symplectic model reduction of Hamiltonian systems on nonlinear manifolds and approximation with weakly symplectic autoencoder." SIAM Journal on Scientific Computing 45.2 (2023): A289-A311.
- Peng, Liqian, and Kamran Mohseni. "Symplectic model reduction of Hamiltonian systems." SIAM Journal on Scientific Computing 38.1 (2016): A1-A27.