# Summary

Scientific computing has become an indispensible tool for many disciplines like biology, engineering and physics. It is, for example, useful in (i) establishing connections between theories and experiments, (ii) making predictions without relying on physical apparatuses and (iii) aiding in adjusting parameters in the design of large-scale projects such as fusion reactors. 

In practice scientific computing involves solving partial differential equations, usually on supercomputers; and this can be very expensive. Scientists have long tried to reduce the cost required to solve these equations by various methods. In this work we focus on data-driven reduced order modeling to do so. This approach in practice often means employing machine learning techniques such as neural networks to process data that come from simulations.

Neural networks have shown enormous potential in various fields, but their application in reduced order modeling is still in its infancy. This can be seen by researchers often neglecting properties in their algorithms that have been observed to be important in traditional scientific computing. These properties often pertain to the structure of the solution of the differential equations.

From theoretical and empirical results we know such structure is often indispensible when performing simulations and should be paid attention to, and structure preservation serves as the main purpose of this dissertation. To this end we design new neural network architectures that preserve structure. *Geometric* has traditionally been used as a synonym for structure-preserving and we will therefore adopt the name *geometric machine learning* as an umbrella term for the ideas introduced in this work. 

Geometric machine learning and neural networks are not a novel concept and other researches have proposed architectures that preserve structure before. There are however many new instances of geometric neural networks presented here that are original work. This dissertation is structured into four main parts.

In Part I we give background information that does not constitute novel work, but lays the foundation for the coming chapters. This first part includes a basic introduction into the theory of Riemannian manifolds, a basic discussion of structure preservation and a short explanation of data-driven reduced order modeling.

Part II discusses a novel optimization framework that generalizes existing neural network optimizers, such as the Adam optimizer and the BFGS optimizer, to manifolds. These new optimizers were necessary to enable the training of a new neural network architecture which we call *symplectic autoencoders* (SAEs).

Part III finally introduces various special neural network layers and architectures. Some of them, like *SympNets* and the *multi-head attention layer*, are well-known , but others like SAEs, *volume-preserving attention* and the *linear symplectic transformer* are new.

In Part IV we give some results based on the new architectures and show how they compare to existing approaches. Most of these applications pertain to applications from physics; but to demonstrate the efficacy of the new optimizers we resort to a classical problem from image classification to show that geometric machine learning can also find applications in fields outside of scientific computing. For all examples that we show, our new architectures exhibit a clear improvement in terms of speed or accuracy over existing architectures. We show one example where we obtain a speed-up of a factor of 1000 with SAEs and transformers.

```@raw latex
\clearpage
```