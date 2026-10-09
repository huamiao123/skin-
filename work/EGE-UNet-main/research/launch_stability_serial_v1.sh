#!/usr/bin/env bash
set -u

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MEDSEG_STABILITY_START_AT=dual_difflr_seed43

log_file="results/stability_resume_20260929_console.log"
printf 'START_UTC=%s\n' "$(date -u +%FT%TZ)" >> "$log_file"
python -u research/run_stability_serial_v1.py >> "$log_file" 2>&1
status=$?
printf 'END_UTC=%s EXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$status" >> "$log_file"
exit "$status"
