#!/usr/bin/env bash
set -u

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

log_file="results/stability_ege_baseline_seed43_console.log"
printf 'QUEUED_UTC=%s waiting_for=medseg_stability\n' "$(date -u +%FT%TZ)" >> "$log_file"
while tmux has-session -t medseg_stability 2>/dev/null; do
    sleep 30
done

if ! rg -q 'FINISHED: dual_scf_seed43' results/stability_resume_20260929_console.log; then
    printf 'SKIPPED_UTC=%s reason=upstream_seed43_queue_incomplete\n' "$(date -u +%FT%TZ)" >> "$log_file"
    exit 1
fi

printf 'START_UTC=%s\n' "$(date -u +%FT%TZ)" >> "$log_file"
python -u research/run_ege_baseline_seed43.py >> "$log_file" 2>&1
status=$?
printf 'END_UTC=%s EXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$status" >> "$log_file"
exit "$status"
