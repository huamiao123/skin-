#!/bin/bash
echo "[watcher] waiting for SCAUG to finish..."
while true; do
  if grep -q "EGE-DUAL SCAUG COMPLETE" /root/experiments/logs/ege_dual_scaug_run.log 2>/dev/null; then
    echo "[watcher] scaug done, launching SCF"
    setsid bash /root/experiments/scripts/run_ege_dual_scf.sh > /root/experiments/logs/ege_dual_scf_run.log 2>&1 &
    break
  fi
  if ! pgrep -f "ege_dual_difflr_scaug_config" > /dev/null && [ -s /root/experiments/logs/ege_dual_scaug_run.log ]; then
    echo "[watcher] scaug exited, launching SCF anyway"
    setsid bash /root/experiments/scripts/run_ege_dual_scf.sh > /root/experiments/logs/ege_dual_scf_run.log 2>&1 &
    break
  fi
  sleep 300
done
echo "[watcher] SCF launched"
