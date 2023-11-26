# The Existence-And-Uniqueness Theorem

In order to proof the existence-and-uniqueness theorem we first need another theorem, the **Banach fixed-point theorem** for which we also need another definition. 

__Definition__: A **contraction mapping** is a map ``T:\mathbb{R}^N\to\mathbb{R}^N`` for which there exists ``q\in[0,1)`` s.t. ``\forall{}x,y\in\mathbb{R}^N,\,||T(x)-T(y)||\leq{}q||x-y||``.

__Theorem (Banach fixed-point theorem)__: Every **contraction mapping** ``T`` admits a unique fixed point ``x^*`` (i.e. a point ``x^*`` s.t. ``F(x^*)=x^*``) and this point can be found by taking an arbitrary point ``x_0\in\mathbb{R}^N`` and taking the limit ``\lim_{n\to\infty}T^n(x_0)``.

__Proof (Banach fixed-point theorem)__: Take an arbitrary point ``x_0\in\mathbb{R}^N`` and consider the sequence ``(x_n)_{n\in\mathbb{N}}`` with ``x_n:=T^n(x_0)``. Then it holds that (for ``m>n``): 
```math
\begin{aligned}
|x_m - x_n|   & \leq  |x_m - x_{m-1}| + |x_{m-1} - x_{m-2}| + \cdots + |x_{m-(m-n+1)}-x_{n}| \\
                & =     |x_{n+(m-n)} - x_{n+(m-n-1)}| + \cdots + |x_{n+1} - x_n| \\
                & \leq \sum_{i=0}^{m-n-1}q^i|x_{n+1} - x_n| \\
                & \leq \sum_{i=0}^{m-n-1}q^iq^n|x_1 - x_0| \\
                & = q^n|x_1 -x_0|\sum_{i=1}^{m-n-1}q^i,
\end{aligned}
```
where we have used the triangle inequality in the first line. If we now let ``m`` on the right-hand side first go to infinity then we get 
```math
|x_m-x_n|     & \leq q^n|x_1 -x_0|\sum_{i=1}^{\infty}q^i
                & =q^n|x_1 -x_0| \frac{1}{1-q},
```  
proofing that the sequence is Cauchy. Because ``\mathbb{R}^N`` is a complete metric space we get that ``(x_n)_{n\in\mathbb{N}}`` is a convergent sequence. We call the limit of this sequence ``x^*``. This completes the proof of the Banach fixed-point theorem. 
