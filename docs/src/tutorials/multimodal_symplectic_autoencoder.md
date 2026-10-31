# Multi-Modal Symplectic Autoencoders

A [`MultiModalSymplecticAutoencoder`](@ref) maps several descriptions of one phase-space state into
one latent space ``\xi = (\tilde{q}, \tilde{p}) \in \mathbb{R}^{2n}``. Each description, or
*modality*, has its own frontend layer, and all frontends share one symplectic core. This page
builds such a network for a 1D1V system with three modalities, and trains it in three stages.
The script `scripts/reproduction/multimodal_symplectic_autoencoder.jl` runs every step of this
page in `Float32` and in `Float64`, and asserts the claim of each stage.

!!! warning "The dataset is synthetic"
    The snapshots on this page are built from two spatial modes with fixed velocity profiles,
    plus noise. They test that the pipeline runs and that each stage does what it claims. They
    say nothing about the physics: whether a shared latent space reproduces a Vlasov or a
    particle simulation better than a linear reduced basis is the question of a separate
    benchmark on Landau damping and the two-stream instability, not of this page.

## The architecture

The network is a `NamedTuple` of frontends and a core:

```julia
using GeometricMachineLearning
using Random: Xoshiro

frontends = (grid = BilinearStiefelLayer(16, 16, 2, 2),
    dlra = LowRankStiefelLayer(16, 16, 2, 2; r_out = 4),
    particles = CanonicalSlotLayer(8, 1, 4))
core = WeightTiedSymplecticAutoencoder(8, 4; n_blocks = 1, n_layers = 2,
    n_ambient_layers = 0, sympnet_upscale = 4)
nn = NeuralNetwork(MultiModalSymplecticAutoencoder(frontends, core), CPU(), Float64;
    rng = Xoshiro(42))
```

Each frontend maps its modality to ``2N = 8`` frontend coordinates, the input dimension of the
core, and the core maps them to ``2n = 4`` latent coordinates.

- **The grid frontend**, [`BilinearStiefelLayer`](@ref)`(Nx, Nv, mx, mv)`, takes a field
  ``F \in \mathbb{R}^{N_x \times N_v}``, splits it into its even and odd parts in ``v``, and
  projects both onto Stiefel bases ``\Phi_x`` and ``\Phi_v``. A spline modality is the same layer
  with `metric_factors = (Rx, Rv)`, the factors of the mass matrices of the spline bases.
- **The DLRA frontend**, [`LowRankStiefelLayer`](@ref), takes a factored state ``(X, S, V)``
  with ``F = X S V^T`` and gives the coordinates that the grid frontend gives for ``F``, without
  assembling ``F``.
- **The particle frontend**, [`CanonicalSlotLayer`](@ref)`(M, d, NSlots)`, takes ``M`` particles
  in ``d`` dimensions, sorts them, and keeps their centre of mass and ``N_{\mathrm{slots}} - 1``
  relative modes in a Stiefel basis ``\Phi_{\mathrm{rel}}``.

The latent coordinates are canonical for the structure ``\mathbb{J}_{2N}`` that each frontend
declares on its coordinates. The two parity frontends are linear in ``F``, so they are not Poisson
maps for the Vlasov bracket; the docstring of [`BilinearStiefelLayer`](@ref) says more.

[`encode`](@ref)`(nn, modality, x)` applies a frontend and the core's encoder,
[`decode`](@ref)`(nn, modality, ξ)` applies the core's decoder and the inverse of a frontend, and
[`translate`](@ref)`(nn, from, to, x)` is `decode(nn, to, encode(nn, from, x))`.

The core is a [`WeightTiedSymplecticAutoencoder`](@ref): its encoder is the exact inverse of its
decoder layers, applied to the same parameters. Its decode–encode is therefore the projection of
its last [`PSDLayer`](@ref), whatever the gradient layers between, and the reconstruction error of
the network is set by its linear parts: the bases of the frontend and the symplectic lift of the
core.

## The data

The script draws ``B = 8`` snapshots in smoke mode and ``B = 32`` in full mode, all with
`Random.Xoshiro(42)`. A snapshot is ``F_b = f_0 + \delta F_b``: a background ``f_0``, even in
``v``, and a fluctuation ``\delta F_b`` of two spatial modes, each with an even and an odd velocity
profile, plus noise of size ``10^{-3}``. The DLRA snapshots are the rank-4 truncated SVDs of the
fluctuations. The frontends are linear, so the network sees the fluctuations, and the background
returns in Stage III as the reference state.

## Stage I: subspaces from the SVD

[`solve!`](@ref) sets the linear parts of the network from singular value decompositions, with
no training:

```julia
using GeometricMachineLearning: params
using GeometricOptimizers: solve!

ps = params(nn)
solve!(nn.architecture.frontends.grid, ps.frontends.grid, δF)   # Φx and Φv of the grid
copyto!(parent(ps.frontends.dlra.Φx), parent(ps.frontends.grid.Φx))
copyto!(parent(ps.frontends.dlra.Φv), parent(ps.frontends.grid.Φv))

psd = NeuralNetwork(PSDArch(8, 4), CPU(), Float64)
solve!(psd, Z)                                                    # the cotangent lift of Z
copyto!(parent(ps.core[end].weight), parent(params(psd)[1].weight))
```

Here `δF` holds the fluctuations, of size `(16, 16, B)`, and `Z` holds their frontend
coordinates, one column per snapshot. The script asserts that the projection error

```math
\varepsilon_I = \max_b \frac{\|F_b - (f_0 + \Pi(F_b - f_0))\|}{\|F_b\|},
```

with ``\Pi`` the decode–encode of the grid modality through the whole network, is below
``10^{-2}``. Without Stage I it is about ``0.15``.

## Stage II: joint training

Stage II trains all modalities together. Its loss is the sum of three terms over the batch: the
[`CharbonnierRelativeLoss`](@ref) of the grid modality, the [`CrossModalLoss`](@ref) from the
DLRA modality to the grid modality, and the relative squared error of the particle states that
`translate(nn, :grid, :particles, δF)` gives. The particle term translates rather than encodes:
the particle frontend sorts its input, and a decoded particle state need not be in the order that
the sort keeps.

The particle states of the synthetic dataset are built with a relative basis that is not the
network's, so Stage II has to learn ``\Phi_{\mathrm{rel}}``. The optimizer steps the Stiefel leaves
with `FactoredStiefelAdam`, a direct tangent-space Adam of `GeometricOptimizers` that keeps each
leaf on the manifold, and all other leaves with
[`Adam`](@extref GeometricOptimizers GeometricOptimizers.Adam):

```julia
using GeometricOptimizers: TrainingOptimizer, CompositeMethod, FactoredStiefelAdam, Adam,
                           optimization_step!
using Zygote: Zygote

optimizer = TrainingOptimizer(params(nn);
    algorithm = CompositeMethod(; manifold = FactoredStiefelAdam(), array = Adam()),
    linesearch = 1e-3)
for step in 1:400
    dp = Zygote.gradient(p -> loss(nn.model, p), params(nn))[1]
    optimization_step!(params(nn), optimizer, dp)
end
```

See [`TrainingOptimizer`](@extref GeometricOptimizers GeometricOptimizers.TrainingOptimizer) and
[`CompositeMethod`](@extref GeometricOptimizers GeometricOptimizers.CompositeMethod). The script
prints the loss after every step and asserts two things: that the loss of the last step is at
most half the loss after Stage I, and that every
[`StiefelManifold`](@extref GeometricOptimizers GeometricOptimizers.StiefelManifold) leaf
``Y`` satisfies ``\|Y^T Y - I\|_F \le 50\,\mathrm{eps}(T)``. Adam is not a descent method, so the
loss need not fall at every step, and the script does not assert that it does. Euclidean Adam on
the Stiefel leaves breaks the second assertion.

## Stage III: the reduced system

Stage III integrates the reduced system of a Hamiltonian on the frontend coordinates with
[`HRedSys`](@ref). The core's [`encoder`](@ref) and [`decoder`](@ref) reduce the ``2N = 8``
frontend coordinates to ``2n = 4``, and the reference state `q_ref, p_ref` is the frontend
coordinates of the background ``f_0``:

```julia
using GeometricIntegrators: ImplicitMidpoint, StateVariable

z_ref = frontend_coordinates(f0)
z0 = frontend_coordinates(F0)
ics = (q = StateVariable(z0[1:4]), p = StateVariable(z0[5:8]))
rs = HRedSys(8, 4, encoder(nn), decoder(nn), v!, f!, h, (0.0, 2.0), 2.0 / 20, ics;
    integrator = ImplicitMidpoint(), q_ref = z_ref[1:4], p_ref = z_ref[5:8],
    pullback_mode = :vjp)
sol = integrate_reduced_system(rs)
```

Here `frontend_coordinates(F)` is the output `(q, p)` of the grid frontend as one vector, and
`v!`, `f!` and `h` are the vector fields and the energy of a pendulum on each coordinate,
``H(q, p) = \frac{1}{2}|p|^2 + \sum_i (1 - \cos q_i)``. The script asserts two things:

- The Hamiltonian is not quadratic, so the implicit midpoint rule conserves it only to second
  order in the time step. Over the interval ``(0, 2)``, the largest error of the reduced energy in
  20 steps is between 3 and 5 times the error in 40 steps.
- The decoded initial field ``f_0 + \mathrm{field}(\mathrm{decode}(\mathrm{encode}(z_0)) - z_{\mathrm{ref}})``
  equals ``F_0`` to ``\varepsilon_I \|F_0\|``. Without `q_ref, p_ref`, the background is not in
  the range of the rank-``2n`` decoder, and the relative error is about 1.

The script integrates a fixed latent Hamiltonian and trains no latent dynamics. To fine-tune a
latent integrator on decoded trajectories, use [`IsometricRolloutLoss`](@ref).

## Run time

The script prints the time from the end of `using` to the end of Stage III, compilation included,
and asserts a limit on it: 60 seconds in smoke mode and 180 seconds in full mode. Almost all of
the time is compilation: the first gradient of the Stage II loss and the first integration of each
precision.
