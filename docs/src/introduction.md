# Introduction and Outline

One of the most popular books on neural networks [goodfellow2016deep](@cite) introduces the term *machine learning* the following way: "The difficulties faced by systems relying on hard-coded knowledge suggest that [artificial intelligence] systems need the ability to acquire their own knowledge, by extracting patterns from raw data. This capability is known as machine learning." The many success stories of deep neural networks such as ChatGPT [achiam2023gpt](@cite) have shown that abandoning hard-coded knowledge in favour of extracting patterns can yield enormous improvement for many applications. In numerics the story is a different one however. Hard coding of certain properties into an algorithm has proved indispensible for many numerical applications. The introduction to one of the canonical references on geometric numerical integration [hairer2006geometric](@cite) contains the sentence: "It turned out that the preservation of geometric properties of the flow not only produces an improved qualitative behaviour, but also allows for a more accurate long-time integration than with general-purpose methods" and "preservation of geometric properties" means *hard-coded physical information* here.

Researchers have realized very eraly that systems that work for image recognition, natural language processing and other purely data-driven tasks may not be suitable to treat problems from physics [psichogios1992hybrid](@cite). Scientific machine learning [baker2019workshop](@cite), which can be roughly defined as an application of machine learning techniques to science and engineering, has however much too often neglected the preservation of geometric properties that has proved to be so important in traditional numerics. An ostensible solution is offered by so-called physics-informed neural networks (PINNS), whose eponymous paper [raissi2019physics](@cite) is one of the most-cited in scientific machine learning. The authors write: "Coming to our rescue, for many cases pertaining to the modeling of physical and biological systems, there exists a vast amount of prior knowledge that is currently not being utilized in modern machine learning practice." It is stated that PINNS "[enrich] deep learning with the longstanding developments in mathematical physics;" one should however add that they also ignore *longstanding developments in numerics*, like preserving the geometric properties which are observed to be crucial in [hairer2006geometric](@cite).

What this work aims at doing is not "to set the foundations for a new paradigm" [raissi2019physics](@cite), but rather to show that it makes sense to imbue neural networks with specific structure and one should to do this whenever possible. In this regard this work is much more closely related to traditional numerics than to neural network research as we try to design problem-specific algorithms rather than "universal approximators" [hornik1989multilayer](@cite). The *structure-preserving neural networks* in this work are never fundamentally new architectures but build on existing neural network designs [vaswani2017attention, jin2020sympnets](@cite) or more classical methods [peng2016symplectic](@cite). We design neural networks that have a specific structure encoded in them (modeling part) and then make their behavior reflect information found in data (machine learning part). We refer to this as *geometric machine learning*.

```@example
Main.include_graphics("tikz/gml_venn"; width = .6, caption = "Geometric machine learning (GML) like traditional geometric numerical integration (GNI) and other structure-preserving numerical methods aims at building models that share properties with the analytic solution of a differential equation.") # hide
```

In this picture we visualize that geometric machine learning aims at constructing so-called structure-preserving mappings that are ideally close to the analytic solution. *Structure-preserving* here means that the model shares properties with the analytic solution of the underlying differential equation. In this work the most important of these properties are *symplecticity* and *volume preservation*, but this may extend to others such as the null space of certain operators [arnold2006finite](@cite).

For us the biggest motivation for geometric machine learning comes from *data-driven reduced order modeling*. There we want to find cheap representations of so-called *full order models* of which we have data available. This is the case when solving parametric partial differential equations (PPDEs) for example. In this case we can solve the full order model for a few parameter instances and then build a cheaper representation of the full model (a so-called *reduced model*) with neural networks. This can bring dramatic speed-ups in performance. 

Closely linked to the research presented here is the development of a software package written in `Julia` called `GeometricMachineLearning`. Throughout this work we will demonstrate concepts such as neural network architecture and (Riemannian) optimization by using `GeometricMachineLearning`. Most sections contain a header **Library Functions** that explains types and functions in `GeometricMachineLearning` that pertain to the text in that section. All the code necessary to reproduce the results is included in the text and does not have any specific hardware requirements. Except for training some of the neural networks (which was done on an NVIDA Geforce RTX 4090 [rtx4090](@cite)) all the code snippets were run on CPU (via GitHub runners [actions](@cite)).

This dissertation is structures into four main parts: background information, an explanation of the optimizer framework used for training neural networks, a detailed explanation of the various neural network layers and architectures that we use and a number of examples of how to use `GeometricMachineLearning`. We explain the content of these parts in some detail[^1].

[^1]: In addition there is also an appendix that provides more implementation details.

## Background Information

The background material, which does not include any original work, covers all the prerequisites for introducing our new optimizers in Part II. In addition it introduces some basic functionality of `GeometricMachineLearning`. It contains (among others) the following sections:
- [Concepts from general topology](@ref "Basic Concepts from General Topology"): here we introduce topological spaces, closedness, compactness, countability and Hausdorffness amongst others. These concepts are prerequisites for defining manifolds.
-  [General theory on manifolds](@ref "(Matrix) Manifolds"): we introduce manifolds, the preimage theorem and submersion theorem. These theorems will be used to construct manifolds; the preimage theorem is used to give structure to the [Stiefel](@ref "The Stiefel Manifold") and the [Grassmann manifold](@ref "The Grassmann Manifold"), and the immersion theorem gives structure to the [solution manifold](@ref "The Solution Manifold") which is used in reduced order modeling.
- [Riemannian manifolds](@ref "Riemannian Manifolds"): for optimizing on manifolds we need to define a metric on them, which leads to *Riemannian manifolds*. We introduce *geodesics* and the *Riemannian gradient* here.
- [Homogeneous spaces](@ref "Homogeneous Spaces"): homogeneous spaces are as special class of manifolds to which our *generalized optimizer framework* can be applied. They trivially include all Lie groups and spaces like the [Stiefel](@ref "The Stiefel Manifold"), the [Grassmann manifold](@ref "The Grassmann Manifold") and the "homogeneous space of positions and orientations" [bon2024optimal](@cite).
- [Global tangent spaces](@ref "Global Tangent Spaces"): homogeneous spaces allow for identifying for an invariant representation of all tangent spaces which we call *global tangent spaces*[^2]. 
- [Geometric structure](@ref "Symplectic Systems"): structure preservation takes a prominent role in this dissertation. In general it *structure* refers to some property that the analytic solution of a differential equation has. Here we discuss *symplecticity* and [*volume preservation*](@ref "Divergence-Free Vector Fields") in detail.
- [Reduced order modeling](@ref "Reduced Order Modeling"): reduced order modeling serves as a motivation for most of the architectures introduced here. 

[^2]: These spaces are also discussed in [bendokat2020grassmann, bendokat2021real](@cite).

## The Optimizer Framework

One of the central parts of this dissertation is an *optimizer framework* that allows the generalization of existing optimizers such as Adam [kingma2014adam](@cite) and BFGS [wright2006numerical; Chapter 6.1](@cite) to homogeneous spaces in a consistent way. This part contains the following sections:
- [Neural Network Optimizers](@ref): here we introduce the concept of a neural network optimizer and discuss the modifications we have to make in order to generalize them to homogeneous spaces.
- [Retractions](@ref): an important concept in manifold optimization are retractions [absil2008optimization](@cite). We introduce them in this section, discuss how they can be constructed for homogeneous spaces and show the two examples of the *geodesic retraction* and the *Cayley retraction*.
- [Parallel Transport](@ref): whenever we have an optimizer that contains momentum terms we need to *transport* these momenta. In this section we explain how this can be done straightforwardly when dealing with homogeneous spaces. 
- [Optimizer methods](@ref "Standard Neural Network Optimizers"): in this section we introduce simple optimizers such as the *gradient optimizer*, the *momentum optimizer* and *Adam* and show how to generalize them to our setting. Due to its increased complexity the BFGS optimizer gets [its own section](@ref "The BFGS Optimizer").

```@example
Main.include_graphics("tikz/tangent_vector"; caption = raw"Weights can be put on manifolds to achieve structure preservation or improved stability.") # hide
```

## Special Neural Network Layers and Architectures

In here we first discuss specific neural network layers and then architectures. A neural network architecture is always a composition of many neural network layers that is designed for a specific task.

Special neural network layers include:
- [SympNet layers](@ref "SympNet Layers"): *symplectic neural networks* (SympNets) [jin2020sympnets](@cite) are special neural networks that are *universal approximators in the class of canonical symplectic maps.* SympNet layers comprise three different types: linear layers, activation layers and gradient layers. All these are introduced here.
- [Volume-preserving layers](@ref "Volume-Preserving Feedforward Layer"): the volume-preserving layers presented here are inspired by linear and activation SympNet layers. They slightly differ from other approaches with the same aim [bajars2023locally](@cite).
- [Attention layers](@ref "The Attention Layer"): many fields in neural network research have seen big improvements due to *attention mechanisms* [vaswani2017attention, patwardhan2023transformers, dosovitskiy2020image](@cite). Here we introduce this mechanism (which is a neural network layer) and also discuss how to make it volume-preserving [brantner2024volume](@cite). It serves as a basis for [multihead attention](@ref "Multihead Attention") and [linear symplectic attention](@ref "Linear Symplectic Attention").

Special neural network architectures include:
- [Symplectic autoencoders](@ref "The Symplectic Autoencoder"): the *symplectic autoencoder* constitutes one of the central elements of this dissertation. It offers a way of flexibly performing nonlinear model order reduction for Hamiltonian systems. In this section we explain its architecture and how it is implemented in `GeometricMachineLearning` in detail.
- [SympNet architectures](@ref "SympNet Architecture"): based on SympNet layers (which are simple building blocks) one can construct two main types of architectures which are called ``LA``-SympNets and ``G``-SympNets. We explain both here.
- [Volume-preserving feedforward neural networks](@ref "Volume-Preserving Feedforward Neural Network"): based on ``LA``-SympNets we build *volume-preserving feedforward neural networks* that can learn arbitrary volume-preserving maps. 
- [Transformers](@ref "Standard Transformer"): transformer neural networks have revolutionized many fields in machine learning like natural language processing [vaswani2017attention](@cite) and image recognition [dosovitskiy2020image](@cite). We discuss them here and further imbue them with structure-preserving properties to arrive at [volume-preserving transformers](@ref "Volume-Preserving Transformer") and [linear symplectic transformers](@ref "Linear Symplectic Transformer").

We note that SympNets, volume-preserving feedforward neural networks and the three transformer types presented here belong to a category of [neural network integrators](@ref "Neural Network Integrators"), which are used in the online stage of reduced order modeling.

## Examples

In this part we demonstrate the neural network architectures implemented in `GeometricMachineLearning` with a few examples:
- [SympNets](@ref "SympNets with `GeometricMachineLearning`"): this serves as an introductory example into using `GeometricMachineLearning` and does not contain any new results. It simply shows how to use SympNets to learn the flow of a harmonic oscillator.
- [Symplectic autoencoders](@ref "Symplectic Autoencoders and the Toda Lattice"):
- [Image classification](@ref "MNIST Tutorial"): Here we perform image classification for the MNIST dataset [deng2012mnist](@cite) with vision transformers and show that manifold optimization can enable convergence that would otherwise not be possible.
- [The Grassmann manifold in neural networks](@ref "Example of a Neural Network with a Grassmann Layer"):
- [Different volume-preserving attention mechanisms](@ref "Comparing Different `VolumePreservingAttention` Mechanisms"):
- [Linear Symplectic Transformer](@ref linear_symplectic_transformer_tutorial): 
- [Adjusting the loss function](@ref "Adjusting the Loss Function"): 

## Associated Papers

The following papers have emerged in connection with the development of `GeometricMachineLearning`:
1. In [brantner2023generalizing](@cite) a new class of optimizers for *homogeneous spaces*, a category that includes the Stiefel manifold and the Grassmann manifold, is introduced.
2. In [brantner2023symplectic](@cite) we introduced a new neural network architectures that we call *symplectic autoencoders*. This is capable of performing non-linear Hamiltonian model reduction. During training we use the optimizers introduced in [brantner2023generalizing](@cite).
3. In [brantner2024volume](@cite) we introduce a new neural network architecture that we call *volume-preserving transformers*. This is a structure-preserving version of the *standard transformer* [vaswani2017attention](@cite) for which all components have been made volume preserving. As application we foresee the *online phase* in reduced order modeling.

In addition there are new results presented in this work that have not been written up as a separate paper:
4. We show how the [Grassmann manifold can be included into a neural network](@ref "Example of a Neural Network with a Grassmann Layer") and construct a loss based on the Wasserstein distance to approximate a nonlinear space from which we can then sample.
5. Similar to the volume-preserving transformer [brantner2024volume](@cite) we introduce a [*linear symplectic transformer*](@ref "Linear Symplectic Transformer") that preserves a symplectic product structure and is also foreseen to be used in reduced order modeling.