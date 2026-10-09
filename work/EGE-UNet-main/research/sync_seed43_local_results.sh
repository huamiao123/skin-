#!/usr/bin/env bash
set -u

local_root=/home/featurize/medseg_seed43_local
mirror_root=/home/featurize/work/EGE-UNet-main/results/seed43_local_mirror_20260929
mkdir -p "$mirror_root/results" "$mirror_root/logs"

sync_once() {
    rsync -a --delete "$local_root/results/" "$mirror_root/results/" &&
    rsync -a --delete "$local_root/logs/" "$mirror_root/logs/" &&
    date -u +'%FT%TZ' > "$local_root/logs/last_nfs_mirror_utc.txt"
}

while tmux has-session -t medseg_stability 2>/dev/null ||
      tmux has-session -t medseg_baseline43_queued 2>/dev/null; do
    sync_once || true
    sleep 600
done
sync_once
