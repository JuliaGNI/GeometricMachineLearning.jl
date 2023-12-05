# Optimization for Neural Networks 

Optimization for neural networks is (almost always) some variation on gradient descent. The most basic form of gradient descent is a discretization of the *gradient flow equation*:
```math
\dot{\theta} = -\nabla_\theta{}L,
```
by means of a Euler time-stepping scheme: 
```math
\theta^{t+1} = \theta^{t} - h\nabla_{\theta^{t}}L,
```
where $\eta$ (the time step of the Euler scheme) is referred to as the *learning rate*
<<<<<<< HEAD
This equation can easily be generalized to [manifolds](../manifolds/manifolds.md) by replacing the *Euclidean gradient* $\nabla_{\theta^{t}L}$ by a *Riemannian gradient* $-h\mathrm{grad}_{\theta^{t}}L$ and addition by $-h\nabla_{\theta^{t}}L$ with a [retraction](../optimizers/manifold_related/retractions.md) by $-h\mathrm{grad}_{\theta^{t}}L$.
=======
This equation can easily be generalized to [manifolds](manifolds.md) by replacing the *Euclidean gradient* $\nabla_{\theta^{t}L}$ by a *Riemannian gradient* $-h\mathrm{grad}_{\theta^{t}}L$ and addition by $-h\nabla_{\theta^{t}}L$ with a [retraction](retraction.md) by $-h\mathrm{grad}_{\theta^{t}}L$.
>>>>>>> 9daa61fb1fc0177fc30f51b3df504196a03c688a
