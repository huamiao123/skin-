#!/bin/bash
echo "[watcher] waiting for EGE-Dual-difflr to finish..."
while true; do
  if grep -q "EGE-DUAL DIFFLR TRAINING COMPLETE" /root/ege_dual_difflr_run.log 2>/dev/null; then
    echo "[watcher] difflr done, launching PTU-Dual"
    setsid bash /root/run_ptu_dual.sh > /dev/null 2>&1 &
    break
  fi
  if ! pgrep -f "ege_dual_difflr_config" > /dev/null && [ -s /root/ege_dual_difflr_run.log ]; then
    echo "[watcher] difflr exited, launching PTU-Dual anyway"
    setsid bash /root/run_ptu_dual.sh > /dev/null 2>&1 &
    break
  fi
  sleep 300
done
echo "[watcher] PTU-Dual launched"
