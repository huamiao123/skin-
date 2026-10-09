#!/usr/bin/env bash
set -u

local_root=/home/featurize/medseg_seed43_local/results/rebatch8_20260930
mirror_root=/home/featurize/work/EGE-UNet-main/results/rebatch8_20260930_mirror
mkdir -p "$mirror_root"

sync_once() {
    rsync -a --delete "$local_root/" "$mirror_root/" &&
    date -u +'%FT%TZ' > "$mirror_root/last_sync_utc.txt"
}

while tmux has-session -t medseg_uniform6_20260930 2>/dev/null; do
    sync_once || true
    sleep 600
done
sync_once
