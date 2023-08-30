# Global Sections

The set of functions in `global_sections.jl` is closely related to the ones in `retractions.jl`. For simplicitly we freely discuss the Stiefel manifold as an example of a homogeneous space. The terminology here can be easily translated to other homogeneous spaces. 

There are two steps to them: 
1. A mapping from the homogeneous space $\mathcal{M}=G/\sim$ to the associated Lie group: $\mathcal{M}\to{}G$. This is done by a separate struct, called `GlobalSection`.
2. A mapping from $T_Y\mathcal{M}\to\mathfrak{g}^\mathrm{hor}$, where $\frak{g}^\mathrm{hor}$ is the **horizontal component** to the Lie algebra (our **global tangent space representation**). See the section for `retractions.jl` for this. 

## Computing the global section
For the Stiefel manifold, `GlobalSection` takes an element of $Y\in{}St(n,N)\equiv$`StiefelManifold{T}` and returns an instance of `GlobalSection{T, StiefelManifold{T}}`. This represents an element $A\in{}O(N)$ such that $AE = Y$. The application $O(N)\times{}St(n,N)\to{}St(n,N)$ is done with the functions `apply_section!` and `apply_section`.

## Computing the global tangent space representation based on a global section

The function `global_rep` does the second step, so it takes an instance of `GlobalSection` and an element of $T_YSt(n,N)$ (simply represented through a matrix), and then returns an element of $\frak{g}^\mathrm{hor}\equiv$`StiefelLieAlgHorMatrix`.

## Why do we need a `GlobalSection` to compute the global tangent space representation?

For each element $Y\in\mathcal{M}$ we can perform a splitting $\mathfrak{g} = \mathfrak{g}^{\mathrm{hor}, Y}\oplus\mathfrak{g}^{\mathrm{ver}, Y}$, where the two subspaces are the **horizontal** and the **vertical** component of $\mathfrak{g}$ ate $Y$ respectively. For homogeneous spaces: $T_Y\mathcal{M} = \mathfrak{g}\cdot{}Y$, i.e. every tangent space to $\mathcal{M}$ can be expressed through the application of the Lie algebra to the relevant element. The vertical component consists of those elements of $\mathfrak{g}$ which are mapped to the zero element of $T_Y\mathcal{M}$, i.e. $\mathfrak{g}^{\mathrm{ver}, Y} := \mathrm{ker}(\mathfrak{g}\to{}T_Y\mathcal{M})$. The orthogonal complement of $\mathfrak{g}^{\mathrm{ver}, Y}$ is the horizontal component and is referred to by $\mathfrak{g}^{\mathrm{hor}, Y}$. This is naturally isomorphic to $T_Y\mathcal{M}$. 

In `GeometricMachineLearning` we do not deal with all vector space $\mathfrak{g}^{\mathrm{hor}, Y}$, but only with $\mathfrak{g}^{\mathrm{hor}, E}\equiv\mathfrak{g}^\mathrm{hor}\equiv$`StiefelLieAlgHorMatrix`. This means we need another mapping $\mathfrak{g}^{\mathrm{hor}, Y}\to\mathfrak{g}^{\mathrm{hor}}$. `global_rep` is a composition of two mappings: $T_Y\mathcal{M} \to \mathfrak{g}^{\mathrm{hor}, Y} \to \mathfrak{g}^{\mathfrak{hor}}$. The second mapping is done through $V \to \lambda(Y)^{-1}V\lambda(Y)$, where $\lambda(Y)$ is a global section of $Y$, i.e. $\lambda(Y)\in{}G$ and $\lambda(Y)E = Y$.

## Optimization

The output of `global_rep` is then used for all the optimization steps. See e.g. the documentation for `adam_optimizer`.