# Conclusion

```@raw latex
\pagestyle{plain}
```

In this dissertation it was shown how neural networks can be imbued with structure to improve their approximation capabilities when applied to physical systems. In the following we summarize the novelties of this work and give an outlook for how it can be expanded in the future.

## Reduced Order Modeling as Motivation

Most of the work presented in this dissertation is motivated by [data-driven reduced order modeling](@ref "Basic Concepts of Reduced Order Modeling"). This is the discipline of building low-dimensional surrogate models from data that come from high-dimensional full order models. Both the low-dimensional surrogate model and the high-dimensional full order model are described by differential equations. When we talk about *structure-preserving reduced order modeling* we mean that the equation on the low-dimensional space [shares features with the equation on the high-dimensional space](@ref "Hamiltonian Model Order Reduction"). In this work these properties were mainly for the vector field to be [symplectic](@ref "Symplectic Systems") or [divergence-free](@ref "Divergence-Free Vector Fields"). A typical reduced order modeling framework is further divided into two phases:
1. in the *offline phase* we find the low-dimensional surrogate model (reduced representation) and
2. in the *online phase* we solve the equations in the reduced space.

For the offline phase we proposed [symplectic autoencoders](@ref "The Symplectic Autoencoder"), and for the online phase we proposed [volume-preserving transformers](@ref "Volume-Preserving Transformer") and [linear symplectic transformers](@ref "Linear Symplectic Transformer"). In the following we summarize the three main methods that were developed in the course of this dissertation and constitute its main results: symplectic autoencoders, structure-preserving transformers and structure-preserving optimizers.

## Structure-Preserving Reduced Order Modeling of Hamiltonian Systems - The Offline Phase

A central part of this dissertation was the development of [symplectic autoencoders](@ref "The Symplectic Autoencoder") [brantner2023symplectic](@cite). Symplectic autoencoders build on existing approaches of *symplectic neural networks* (SympNets) [jin2020sympnets](@cite) and *proper symplectic decomposition* (PSD) [peng2016symplectic](@cite), both of which *preserve symplecticity.* SympNets can approximate arbitrary canonical symplectic maps in ``\mathbb{R}^{2n},`` i.e.

```math
    \mathrm{SympNet}: \mathbb{R}^{2n} \to \mathbb{R}^{2n},
```
but the input has necessarily the same dimension as the output. PSD can change dimension, i.e.[^0]

[^0]: Here we only show the *PSD encoder* ``\mathrm{PSD}^\mathrm{enc}.`` A complete reduced order modeling framework also need a decoder ``\mathrm{PSD}^\mathrm{dec}`` in addition to the encoder. When we use PSD both of these maps are linear, i.e. can be represented as ``(\mathrm{PSD}^\mathrm{enc})^T, \mathrm{PSD}^\mathrm{dec}\in\mathbb{R}^{2N\times{}2n}.``

```math
    \mathrm{PSD}^\mathrm{enc}: \mathbb{R}^{2N} \to \mathbb{R}^{2n},
```

but is strictly linear. Symplectic autoencoders offer a way of (i) constructing nonlinear symplectic maps that (ii) can change dimension. We used these to reduce a 400-dimensional Hamiltonian system to a two-dimensional one[^1]:

[^1]: ``\bar{H} = H\circ\Psi^\mathrm{dec}_{\theta_2}:\mathbb{R}^2\to\mathbb{R}`` here refers to the *induced Hamiltonian on the reduced space*. \textit{SAE} is short for \textit{symplectic autoencoder}. 

```math
(\mathbb{R}^{400}, H) \xRightarrow{\mathrm{SAE}^\mathrm{enc}} (\mathbb{R}^2, \bar{H}).
```

For this case we observed speed-ups of up to a factor 1000 when a [symplectic autoencoder was combined with a transformer in the online phase](@ref "Symplectic Autoencoders and the Toda Lattice"). We also compared the symplectic autoencoder to a PSD, and showed that the PSD was unable to learn a useful two-dimensional representation.

Like PSD, symplectic autoencoders have the property that they induce a Hamiltonian system on the reduced space. This distinguishes them from "weakly symplectic autoencoders" [buchfink2023symplectic, yildiz2024data](@cite) that only approximately obtain a Hamiltonian system on a restricted domain by using a "physics-informed neural networks" [raissi2019physics](@cite) approach.

We also mention that the development of symplectic autoencoders required generalizing existing neural network optimizers to manifolds[^2]. This is further discussed below.

[^2]: We also refer to optimizers that preserve manifold structure as *structure-preserving optimizers*.


## Structure-Preserving Neural Network-Based Integrators - The Online Phase

For the online phase of reduced order modeling we developed new neural network architectures based on the [transformer](@ref "Standard Transformer") [vaswani2017attention](@cite) which is a neural network architecture that is extensively used in other fields of neural network research such as natural language processing[^3]. We used transformers to build an equivalent of structure-preserving multi-step methods [hairer2006geometric](@cite).

[^3]: The T in ChatGPT [achiam2023gpt](@cite) stands for *transformer*.


The transformer consists of a composition of standard neural network layers and attention layers:
```math
    \mathrm{Transformer}(Z) = \mathcal{NN}_n\circ\mathrm{AttentionLayer}_n\circ\cdots\circ\mathcal{NN}_1\circ\mathrm{AttentionLayer}_1(Z),
```
where ``\mathcal{NN}`` indicates a standard neural network layer (e.g. a multilayer perceptron). The attention layer makes it possible for a transformer to process time series data by acting on a whole series of vectors at once:
```math
     \mathrm{AttentionLayer}(Z) = \mathrm{AttentionLayer}(z^{(1)}, \ldots, z^{(T)}) = [f^1(z^{(1)}, \ldots, z^{(T)}), \ldots, f^T(z^{(1)}, \ldots, z^{(T)})].
```
The attention layer thus *performs a preprocessing step* after which the standard neural network layer ``\mathcal{NN}`` is applied.

In this dissertation we presented two modifications of the standard transformer: the [volume-preserving transformer](@ref "Volume-Preserving Transformer") [brantner2024volume](@cite) and the [linear symplectic transformer](@ref "Linear Symplectic Transformer"). In both cases we modified the attention mechanism so that it is either volume-preserving (in the first case) or symplectic (in the second case). The standard neural network layer ``\mathcal{NN}`` was replaced by a [volume-preserving feedforward neural network](@ref "Volume-Preserving Feedforward Neural Network") or a [symplectic neural network](@ref "SympNet Architecture") [jin2020sympnets](@cite) respectively.

In this dissertation we applied the volume-preserving transformer for [learning the trajectory of a rigid body](@ref "The Volume-Preserving Transformer for the Rigid Body") and the linear symplectic transformer for [learning the trajectory of a coupled harmonic oscillator](@ref linear_symplectic_transformer_tutorial). In both cases our new transformer architecture significantly outperformed the standard transformer. The trajectory modeled with the volume-preserving transformer for instance stays very close to a submanifold which is a level set of the quadratic invariant ``I(z_1, z_2, z_3) = z^2_1 + z^2_2 + z^2_3.`` This is not the case for the standard transformer: it moves away from this submanifold after a few time steps.

## Structure-Preserving Optimizers

Training a symplectic autoencoder requires optimization on manifolds[^4]. The particular manifolds we need in this case are "homogeneous spaces" [frankel2011geometry](@cite). In this dissertation we proposed a new optimizer framework that manages to [generalize existing neural network optimizers to manifolds](@ref "Neural Network Optimizers"). This is done by identifying a [global tangent space representation](@ref "Global Tangent Spaces") and dispenses with the need for a *projection step* as is necessary in other approaches [kong2023momentum, li2020efficient](@cite).

[^4]: This is necessary to preserve the symplectic structure of the neural network.

As was already observed by others [zhang2021orthogonality, kong2023momentum, huang2018orthogonal](@cite) putting weights on manifolds can improve training significantly in contexts other than scientific computing. Motivated by this we show an example of training a vision transformer [dosovitskiy2020image](@cite) on the MNIST data set [deng2012mnist](@cite) to demonstrate the efficacy of the new optimizers. Contrary to other applications of the transformer we do not have to rely on layer normalization [xiong2020layer](@cite) or add connections to [achieve convergent training for relatively big neural networks](@ref "MNIST Tutorial"). We also applied the new optimizers to a neural network that contains weights on the [Grassmann manifold](@ref "The Grassmann Manifold") to be [able to sample from a nonlinear space](@ref "Example of a Neural Network with a Grassmann Layer").

## Outlook

We believe that the topics *structure-preserving autoencoders*, *structure-preserving transformers,* *structure-preserving optimizers* and *structure-preserving machine learning* in general offer great potential for future research. 

Symplectic autoencoders could be used for model reduction of higher-dimensional systems [fresca2022pod](@cite) as well as using them for treating systems that are more general than canonical Hamiltonian ones; these include port-Hamiltonian [van2014port](@cite) and metriplectic [morrison1986paradigm](@cite) systems. Structure-preserving model order reductions for such systems have been proposed [gruber2023energetically, moser2023structure, schulze2023structure, mamunuzzaman2022structure](@cite) but without using neural networks. In the appendix we sketch how symplectic autoencoders could be used for [structure-preserving model reduction of port-Hamiltonian systems](@ref "Using Symplectic Autoencoders for Port-Hamiltonian Systems").

Structure-preserving transformers have shown great potential for learning dynamical systems, but their application should not be limited to that area. Structure-preserving machine learning techniques such as *Hamilton Monte Carlo* [duane1987hybrid](@cite) has been used in various fields such as image classification [cobb2021scaling](@cite) and inverse problems [fichtner2018hamiltonian](@cite) and we believe that the structure-preserving transformers introduced in this work can also find applications in these fields, by replacing the activation function in the attention layers of a vision transformer for example.

Lastly structure-preserving optimization is an exciting field, especially with regards to neural networks. The manifold optimizers introduced in this work can speed up neural network training significantly and are suitable for modern hardware (i.e. GPUs). They are however based on existing neural network optimizers such as Adam [kingma2014adam](@cite) and thus still lack a clear geometric interpretation. By utilizing a more geometric representation, as presented in this work, we hope to be able to find a differential equation describing Adam and other neural network optimizer, perhaps through a variational principle [wibisono2016variational, duruisseaux2022accelerated](@cite). One could also build on the existing optimization framework and use retractions other than the *geodesic retraction* and the *Cayley retraction* [presented here](@ref "Retractions"); an example would be a *QR-based retraction* [sato2019cholesky, gao2024optimization](@cite). This will be left for future work.