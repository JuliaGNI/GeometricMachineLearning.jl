# Lagrangian Neural Network

A *Lagrangian neural network* learns a scalar Lagrangian ``L(q, \dot{q})`` instead of the dynamics
themselves. The dynamics are then whatever that Lagrangian implies, through the Euler–Lagrange
equations

```math
    \frac{d}{dt}\frac{\partial{}L}{\partial\dot{q}} = \frac{\partial{}L}{\partial{}q}.
```

This is the Lagrangian counterpart of the [Hamiltonian neural network](@ref hnn_architecture): there
the network is a Hamiltonian and the dynamics come from a Poisson tensor, here the network is a
Lagrangian and the dynamics come from a variational principle.

## The loss

Expanding the Euler–Lagrange equations along a trajectory gives

```math
    \nabla_{\dot{q}}\nabla_{\dot{q}}L\,\ddot{q}
        + (\nabla_q\nabla_{\dot{q}}L)^T\dot{q}
        = \nabla_qL,
```

so the acceleration the learned Lagrangian predicts is

```math
    \ddot{q} = (\nabla_{\dot{q}}\nabla_{\dot{q}}L)^{-1}
               \left( \nabla_qL - (\nabla_q\nabla_{\dot{q}}L)^T\dot{q} \right),
```

which is what [`GeometricMachineLearning.lagrangian_acceleration`](@ref) computes and what
[`GeometricMachineLearning.LNNLoss`](@ref) compares against the data.

The transpose is not cosmetic. ``\nabla_q\nabla_{\dot{q}}L`` is indexed
``[i, j] = \partial^2L/\partial{}q_i\partial\dot{q}_j``, while the chain rule contracts the *first*
index with ``\dot{q}``. It disappears only for a Lagrangian whose position–velocity coupling happens
to be symmetric, which is why the test for this asserts a closed form whose coupling deliberately is
not.

## Why the derivatives are symbolic

The loss needs the *second* derivative of the network with respect to its input, and then its own
derivative with respect to the network parameters. Taking the inner one with `Zygote` inside the
loss does not work: the parameter gradient then fails with
`MethodError: no method matching getindex(::IdDict{Any, Any})`. The gradient and Hessian are
therefore built as symbolic expressions with
[`SymbolicNeuralNetworks`](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl) and compiled, so
that only one differentiation is left for `Zygote` to do. This is the same route
`hamiltonian_vector_field` takes, and for the same reason.

## Training on positions alone

[`GeometricMachineLearning.LNNLoss`](@ref) needs ``\ddot{q}`` in the data. When only positions are
available, [`GeometricMachineLearning.VariationalMidpointLoss`](@ref) trains the same architecture
through the discrete Euler–Lagrange equations of the *midpoint discrete Lagrangian*

```math
    L_d(q_n, q_{n+1}) = \Delta{}t\,
        L\!\left( \frac{q_n + q_{n+1}}{2}, \frac{q_{n+1} - q_n}{\Delta{}t} \right),
```

which ask that ``D_2L_d(q_n, q_{n+1}) + D_1L_d(q_{n+1}, q_{n+2}) = 0`` along the data. Only the
first derivative of the network is needed for this, because the chain rule through the midpoint is
written out rather than taken with a nested `Zygote.gradient`.

## Library Functions

```@docs
GeometricMachineLearning.LNNLoss
GeometricMachineLearning.lagrangian_acceleration
GeometricMachineLearning._euler_lagrange_acceleration
GeometricMachineLearning.VariationalMidpointLoss
GeometricMachineLearning._discrete_lagrangian_derivatives
```
