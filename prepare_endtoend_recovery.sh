#!/usr/bin/env bash
set -euo pipefail

source_dir=experiments/results/endtoend_spatial
audit_dir=experiments/results/endtoend_spatial_infrastructure_attempt
debug_dir=experiments/results/endtoend_spatial_debug_nonfinite

if [[ -e "$audit_dir" ]]; then
    echo "audit target already exists: $audit_dir" >&2
    exit 1
fi
mv "$source_dir" "$audit_dir"
if [[ -e "$debug_dir" ]]; then
    mv "$debug_dir" "$audit_dir/debug_clean_process_confirmation"
fi
