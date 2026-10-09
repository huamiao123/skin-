#!/usr/bin/env bash
set -u

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir" || exit 1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

log_file="results/rebatch8_20260930/queue_console.log"
printf 'START_UTC=%s\n' "$(date -u +%FT%TZ)" >> "$log_file"
for job in wave_v1_seed43 dual_difflr_seed43 ege_baseline_seed42 wave_v1_seed42 dual_difflr_seed42 dual_scf_seed42; do
    printf 'START_JOB=%s UTC=%s\n' "$job" "$(date -u +%FT%TZ)" >> "$log_file"
    python -u research/run_uniform_valbatch8_six.py "$job" >> "$log_file" 2>&1
    status=$?
    printf 'END_JOB=%s UTC=%s EXIT_CODE=%s\n' "$job" "$(date -u +%FT%TZ)" "$status" >> "$log_file"
    if (( status != 0 )); then
        exit "$status"
    fi
done
printf 'END_UTC=%s EXIT_CODE=0\n' "$(date -u +%FT%TZ)" >> "$log_file"
