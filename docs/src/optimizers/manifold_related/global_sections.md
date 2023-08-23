# Global Sections

The set of functions in `global_sections.jl` is closely related to the one in `retractions.jl`.

There are two steps to them: 
1. A mapping from the homogeneous space $\mathcal{M}=G/\sim$ to the associated Lie group: $\mathcal{M}\to{}G$. This is done by a separate struct, called `GlobalSection`.
2. A mapping from $T_Y\mathcal{M}\to\mathfrak{g}^\mathrm{hor}$, where $\frak{g}^\mathrm{hor}$ is the **horizontal component** to the Lie algebra (our **global tangent space representation**). See the section for `retractions.jl` for this. 

For the Stiefel manifold, `GlobalSection` takes an element of $Y\in{}St(n,N)\equiv$`StiefelManifold{T}` and returns an instance of `GlobalSection{T, StiefelManifold{T}}`. This represents an element $A\in{}O(N)$ such that $AE = Y$. The application $O(N)\times{}St(n,N)\to{}St(n,N)$ is done with the functions `apply_section!` and `apply_section`.

The function `global_rep` does the second step, so it takes an instance of `GlobalSection` and an element of $T_YSt(n,N)$ (simply represented through a matrix), and then returns an element of $\frak{g}^\mathrm{hor}\equiv$`StiefelLieAlgHorMatrix`.

The output of `global_rep` is then used for all the optimization steps.