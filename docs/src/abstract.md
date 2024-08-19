# Abstract

Scientific computing has become an indispensible tool for many disciplines like biology, engineering and physics. It is, for example, useful in (i) establishing connections between theories and experiments, (ii) making predictions without relying on physical apparatuses and (iii) aiding in adjusting parameters in the design of large-scale projects such as fusion reactors. 

In practice scientific computing involves solving partial differential equations, usually on supercomputers; and this can be very expensive. Scientists have long tried to reduce the cost required to solve these equations by various methods. In this work we focus on data-driven reduced order modeling to do so. This approach in practice often means process data coming from simulations with machine learning techniques such as neural networks.

Neural networks have shown enormous potential in various applications, but their application in reduced order modeling is still in its infancy. This can be seen by researchers often neglecting properties in their algorithms that have been observed to be important in traditional scientific computing. These properties often pertain to the structure of the solution of the differential equations.

From theoretical and empirical results we know such structure is often indispensible when performing simulations and should be paid attention to, and structure preservation serves as the main purpose of this dissertation. To this end we design new neural network architectures that preserve structure. *Geometric* has traditionally been used as a synonym for structure-preserving，we will therefore adapt the name *geometric machine learning* as an umbrella term for the ideas introduced in this work. 

Geometric neural networks are not a novel concept and other researches have proposed architectures that preserve structure before. There are however many new instances of geometric neural networks presented here that are original work.

In Part I we give background information that does not constitute novel work, but lays the groundwork for the coming chapters. This first part includes a basic introduction into the theory of Riemannian manifolds, a basic discussion of structure preservation and a short explanation of reduced order modeling.

Part II discusses a novel optimization framework that generalizes existing neural network optimizers, such as the Adam optimizer and the BFGS optimizer, to manifolds. These new optimizers were necessary to enable the training of a new neural network architecture which we call *symplectic autoencoders*.

Part III finally introduces various special neural network layers and architectures. Some of them, like *SympNets* and the *multi-head attention layer*, are well-known , but others like the *volume-preserving attention* and the *linear symplectic transformer* are new.

In Part IV we give some results based on the new architectures and show how they compare to existing approaches. Most of these applications pertain to applications from physics; but to demonstrate the efficacy of the new optimizers we resort to a classical problem from image classification to show that geometric machine learning can also find applications in fields outside of scientific machine learning.

```@raw latex
\clearpage
```