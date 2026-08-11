# GeometricOptimizers Migration

Migrate GeometricMachineLearning’s optimizer backend to the local
`GeometricOptimizers` API with minimal compatibility bridges.

## Completed

- Replaced GML optimizer internals with GeometricOptimizers.
- Updated optimizer and layer tests to the current `Adam` API.
- Added Stiefel and Grassmann global-section, retraction, copying, allocation,
  arithmetic, and section-application bridges.
- Preserved manifold-only `rgrad` handling and Euclidean gradients.
- Fixed one-based optimizer iteration timing and first-step test semantics.
- Made `scripts/inspect_go_migration.sh` safe when its report is untracked.

## Validation

Focused suite passed on August 11, 2026:

```text
test/transformer_related/transformer_optimizer.jl
test/layers/manifold_layers.jl
test/layers/gradient_layer_tests.jl
test/symplectic_autoencoder_tests.jl
test/optimizers/optimizer_convergence_tests/psd_optim.jl
test/optimizers/optimizer_convergence_tests/svd_optim.jl
```

## Local Inspection

Run from the repository root:

```bash
./scripts/inspect_go_migration.sh
./scripts/inspect_go_migration.sh --run-tests
```

Override the sibling checkout or report path when needed:

```bash
GO_ROOT=/path/to/GeometricOptimizers \
REPORT_PATH=/tmp/go-migration-report.txt \
./scripts/inspect_go_migration.sh --run-tests
```

Reports contain local paths and test output; review before sharing.

## Delivery

- Keep generated archives, logs, PDFs, auxiliary files, reports, and unrelated
  data files out of commits.
- Review the diff, run focused tests, commit with contribution attribution,
  and push `use-geometric-optimizers` to `origin`.
