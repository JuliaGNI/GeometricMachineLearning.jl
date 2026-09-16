# `scripts/`

Three directories, each with one purpose.

| directory | holds | how CI runs it |
|:--|:--|:--|
| `verification/` | checks that establish a mathematical claim about the package | in full |
| `reproduction/` | the runs that produced the committed weights and the manual's figures | in smoke mode |
| `utilities/` | what the other two include, plus the standalone tools | not run |

`scripts/runscripts.jl` is the driver, and `.github/workflows/Scripts.yml` is the job that calls
it. Every `.jl` file under `verification/` and `reproduction/` is an entry point and is run, each
in its own process. There is no list of what to run, so a script added to either directory is
gated from the moment it lands — and a file that is *included* rather than run belongs in
`utilities/`, or the driver will try to run it on its own and fail.

```sh
julia --startup-file=no --project=scripts scripts/runscripts.jl verification
julia --startup-file=no --project=scripts scripts/runscripts.jl reproduction
```

## Size constants

A reproduction script trains, and training takes hours. Every constant that costs time is written

```julia
include("../utilities/smoke.jl")
const n_epochs = smoke_size(2048, 2)
```

`smoke_size` returns its second argument when `GML_SMOKE` is set in the environment, which is what
the CI job sets. A smoke run establishes that the script still executes end to end, and nothing
about the result — that is the honest limit of what a runner can check for a job that belongs on a
GPU.

To reproduce a result, run the script with `GML_SMOKE` unset. That is the default.

## What the gate does not cover

`SKIPPED` in `scripts/runscripts.jl` names the scripts the job does not run, with the reason for
each. Both have no CPU branch, and no CI runner has a GPU. Those files are *not* known to work;
they are known to be unreachable from here.
