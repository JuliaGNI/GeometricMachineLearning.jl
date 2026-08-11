#!/usr/bin/env bash

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || {
    printf 'Run this script from inside the GeometricMachineLearning repository.\n' >&2
    exit 1
}

cd "$repo_root" || exit 1

go_root=${GO_ROOT:-"$repo_root/../GeometricOptimizers"}
run_tests=0
case "${1:-}" in
    '') ;;
    --run-tests) run_tests=1 ;;
    *) printf 'Usage: %s [--run-tests]\n' "$0" >&2; exit 2 ;;
esac

timestamp=$(date +%Y%m%d_%H%M%S)
report_path=${REPORT_PATH:-"$repo_root/go_migration_inspection_${timestamp}.txt"}

section() {
    printf '\n===== %s =====\n' "$1"
}

run_command() {
    printf '\n$ %s\n' "$*"
    "$@" 2>&1 || printf '[command exited with status %s]\n' "$?"
}

run_shell() {
    printf '\n$ %s\n' "$1"
    bash -lc "$1" 2>&1 || printf '[command exited with status %s]\n' "$?"
}

search_command() {
    if command -v rg >/dev/null 2>&1; then
        rg "$@"
    else
        pattern=$1
        shift
        grep -R -n -E --exclude-dir=.git "$pattern" "$@"
    fi
}

{
    printf 'GeometricOptimizers migration inspection report\n'
    printf 'Generated: %s\n' "$(date)"
    printf 'Repository: %s\n' "$repo_root"
    printf 'GO sibling: %s\n' "$go_root"

    section 'Environment'
    run_command git rev-parse --abbrev-ref HEAD
    run_command git status --short --branch
    run_command git branch -vv --no-color
    run_command git log -8 --oneline --decorate
    run_command julia --version
    if [[ -f Project.toml ]]; then
        run_command sed -n '1,140p' Project.toml
    fi

    section 'Workplan'
    if [[ -f WORKPLAN.md ]]; then
        run_command sed -n '1,260p' WORKPLAN.md
    else
        printf 'WORKPLAN.md not found\n'
    fi

    section 'Tracked migration diff'
    run_shell 'git diff --no-ext-diff -- src test Project.toml Manifest.toml'

    section 'Changed paths'
    run_shell 'git status --short --untracked-files=all'

    section 'Affected GML source'
    for file in \
        src/utils.jl \
        src/GeometricMachineLearning.jl \
        src/manifolds/stiefel_manifold.jl \
        src/manifolds/grassmann_manifold.jl \
        src/arrays/stiefel_lie_algebra_horizontal.jl; do
        if [[ -f "$file" ]]; then
            printf '\n--- %s ---\n' "$file"
            sed -n '1,430p' "$file"
        fi
    done

    section 'Affected tests'
    for file in \
        test/transformer_related/transformer_optimizer.jl \
        test/transformer_related/multi_head_attention_stiefel_optim_cache.jl \
        test/layers/manifold_layers.jl \
        test/layers/gradient_layer_tests.jl \
        test/symplectic_autoencoder_tests.jl \
        test/optimizers/optimizer_convergence_tests/psd_optim.jl \
        test/optimizers/optimizer_convergence_tests/svd_optim.jl; do
        if [[ -f "$file" ]]; then
            printf '\n--- %s ---\n' "$file"
            sed -n '1,300p' "$file"
        fi
    done

    section 'Optimizer/API references'
    printf '\n$ search_command "global_section|GlobalSection|copyto!|_add!|_rac!|_square!|_div!|_rmul!|AdamOptimizer|AbstractCache|GradientMethod|MomentumMethod|Adam(" src test\n'
    search_command 'global_section|GlobalSection|copyto!|_add!|_rac!|_square!|_div!|_rmul!|AdamOptimizer|AbstractCache|GradientMethod|MomentumMethod|Adam\(' src test || true

    section 'Local GeometricOptimizers source'
    if [[ -d "$go_root" ]]; then
        run_command git -C "$go_root" status --short --branch
        run_command git -C "$go_root" log -8 --oneline --decorate
        for file in \
            src/optimizers/optimizer_methods.jl \
            src/optimizers/optimizer_cache.jl \
            src/optimizers/optimizer_state.jl \
            src/manifold_optimizers/gradient_optimizer.jl \
            src/manifold_optimizers/momentum_optimizer.jl \
            src/manifold_optimizers/adam_optimizer.jl \
            src/global_sections/global_sections.jl \
            src/utils.jl; do
            if [[ -f "$go_root/$file" ]]; then
                printf '\n--- GeometricOptimizers/%s ---\n' "$file"
                sed -n '1,320p' "$go_root/$file"
            fi
        done
    else
        printf 'Local GeometricOptimizers checkout not found at %s\n' "$go_root"
        printf 'Set GO_ROOT=/path/to/GeometricOptimizers and rerun.\n'
    fi

    section 'Generated/untracked artifact summary'
    printf '\n$ git status --short --untracked-files=all | search_command "(^\\?\\?|generated|\\.(zip|aux|fdb_latexmk|fls|log|out|pdf|jld2)$)" -\n'
    git status --short --untracked-files=all | search_command '(^\?\?|generated|\.(zip|aux|fdb_latexmk|fls|log|out|pdf|jld2)$)' - || true

    if [[ "$run_tests" -eq 1 ]]; then
        section 'Focused test results'
        julia_cmd=(julia --project=.)
        tests=(
            test/transformer_related/transformer_optimizer.jl
            test/layers/manifold_layers.jl
            test/layers/gradient_layer_tests.jl
            test/symplectic_autoencoder_tests.jl
            test/optimizers/optimizer_convergence_tests/psd_optim.jl
            test/optimizers/optimizer_convergence_tests/svd_optim.jl
        )
        for test_file in "${tests[@]}"; do
            printf '\n--- %s ---\n' "$test_file"
            "${julia_cmd[@]}" -e "using GeometricMachineLearning; include(\"$test_file\")" 2>&1 || \
                printf '[test exited with status %s]\n' "$?"
        done
    else
        section 'Focused test results'
        printf 'Not run. Re-run with --run-tests to include the six focused test outputs.\n'
    fi
} >"$report_path"

printf 'Inspection report written to:\n%s\n' "$report_path"
printf 'Review it for secrets or private paths before sharing.\n'
