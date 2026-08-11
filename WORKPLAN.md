# GeometricOptimizers Migration Workplan

Migrate GML’s optimizer backend to the current local `GeometricOptimizers`
API. Follow GO semantics; add only the GML bridge required for compatibility.
Documentation is out of scope for this round.

## Completed

- Replaced the old GML optimizer internals with GO.
- Updated optimizer/layer tests to typed `Adam` and explicit `step_size`.
- Added GML GO bridges for Stiefel and Grassmann global sections.
- Matched GO Adam construction, default step sizes, and one-based iteration timing.
- Added `copyto!` for `StiefelLieAlgHorMatrix`.
- Applied `rgrad` only to manifold leaves; preserved Euclidean gradients.
- Routed momentum updates through GO’s `MomentumMethod` API.

## Remaining

1. **Fix the GO bridge shape handling**
   - `StiefelLieAlgHorMatrix.A` is a compressed `SkewSymMatrix`, not a dense matrix.
   - The custom Stiefel and Grassmann `update_section!` methods must use GO's
     `apply_section!` path rather than slicing `expB.A`.
   - Recheck copy/update behavior for CPU and GPU-backed arrays.

2. **Fix the transformer assertion**
   - Verify first-step directions, step sizes, and independent cache/state objects.
   - Confirm whether Gradient and Momentum intentionally match on step one.
   - Replace the stale chained inequality with behavior guaranteed by current GO semantics.

2. **Run focused tests**
   ```text
   test/transformer_related/transformer_optimizer.jl
   test/layers/manifold_layers.jl
   test/layers/gradient_layer_tests.jl
   test/symplectic_autoencoder_tests.jl
   test/optimizers/optimizer_convergence_tests/psd_optim.jl
   test/optimizers/optimizer_convergence_tests/svd_optim.jl
   ```
   Cover Float32/Float64, nested parameter trees, mixed Euclidean/manifold leaves,
   and Stiefel/Grassmann parameters.

3. **Audit GO bridges against `../GeometricOptimizers`**
   - Check global sections, copying, arithmetic helpers, retractions, cache/state
     construction, type adaptation, and iteration ownership.
   - Remove obsolete compatibility code; do not duplicate GO implementations.

4. **Check public API compatibility**
   - Review exports and aliases for `GradientMethod`, `MomentumMethod`, `Adam`,
     `AbstractCache`, `AdamOptimizer`, and `AdamOptimizerWithDecay`.
   - Ensure supported GO types are used and removed GML-only implementations are not.

5. **Review the diff and artifacts**
   - Keep only migration changes; exclude generated docs, archives, logs, PDFs,
     auxiliary files, and unrelated `.jld2` output unless explicitly requested.
   - Do not delete user data without confirmation.

6. **Run broader validation**
   - Run relevant optimizer/manifold test groups, package loading/precompilation,
     and formatting for changed files.
   - Record unrelated pre-existing failures separately.

## Local Inspection Script

When network access is unavailable, run this from the GML repository:

```bash
./scripts/inspect_go_migration.sh
```

To include the focused tests:

```bash
./scripts/inspect_go_migration.sh --run-tests
```

If GO is not in the sibling directory, set `GO_ROOT`; to choose the report path,
set `REPORT_PATH`:

```bash
GO_ROOT=/path/to/GeometricOptimizers \
REPORT_PATH=/tmp/go-migration-report.txt \
./scripts/inspect_go_migration.sh --run-tests
```

The report captures branch/diff state, this workplan, affected GML and GO source,
tests, API references, artifacts, and optional test output. Review it for private
paths or sensitive data before sharing.

## Delivery

- Confirm branch `use-geometric-optimizers` and its upstream.
- Create one or a few focused commits with appropriate attribution.
- Ensure only intended migration changes are committed.
- Push to `origin` and report the commit hash, branch, tests, and remaining artifacts.

## Checkpoint

The latest focused run (August 10, 2026) found three migration failures:

- `transformer_optimizer.jl` reaches its final assertion, but Gradient and
  Momentum produce approximately equal first-step parameters; the chained
  `≉` expression is stale because Momentum's first step has no prior momentum.
- `manifold_layers.jl` and `psd_optim.jl` fail while GO copies a
  `GrassmannLieAlgHorMatrix` into a dense matrix, exposing a representation
  mismatch in the bridge.
- `svd_optim.jl` fails when the custom Stiefel `update_section!` slices the
  compressed `SkewSymMatrix` field as though it were dense.

The inspection script also attempted to use `rg`, which is unavailable in the
current shell; its search/report commands need a portable grep fallback.
