#!/bin/bash
echo "[watcher] waiting for PTU depth chain to finish..."
while true; do
  if grep -q "PTU D6 COMPLETE" /root/ptu_d6_run.log 2>/dev/null; then
    echo "[watcher] PTU chain done, launching Swin-Unet scratch [2,2,6,2]"
    cd /root/Swin-Unet-main
    setsid python3 train_swin_isic.py --scratch --tag scratch_d2262 > /root/swin_unet_scratch_d2262_run.log 2>&1 &
    SWIN_PID=$!
    wait $SWIN_PID
    echo "[watcher] Swin scratch done, launching PTU-Base-12-Symmetric"
    chmod +x /root/run_ptu_sym.sh
    setsid bash /root/run_ptu_sym.sh > /root/ptu_sym_prep.log 2>&1 &
    break
  fi
  if ! pgrep -f "train.py --depth3" > /dev/null && ! pgrep -f "run_ptu_depth_chain" > /dev/null; then
    echo "[watcher] PTU chain exited (check logs), launching Swin anyway"
    cd /root/Swin-Unet-main
    setsid python3 train_swin_isic.py --scratch --tag scratch_d2262 > /root/swin_unet_scratch_d2262_run.log 2>&1 &
    SWIN_PID=$!
    wait $SWIN_PID
    chmod +x /root/run_ptu_sym.sh
    setsid bash /root/run_ptu_sym.sh > /root/ptu_sym_prep.log 2>&1 &
    break
  fi
  sleep 300
done
echo "[watcher] PTU-SYM12 launched"
