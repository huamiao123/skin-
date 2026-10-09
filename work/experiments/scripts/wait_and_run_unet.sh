#!/bin/bash
SWIN_LOG=/root/swin_unet_isic18_run.log
echo "[watcher] waiting for Swin-Unet to finish..."
while true; do
  if grep -q "SWIN-UNET TRAINING COMPLETE" "$SWIN_LOG" 2>/dev/null; then
    echo "[watcher] Swin done, launching U-Net"
    setsid bash /root/run_unet_isic.sh > /root/unet_isic18_run.log 2>&1 &
    break
  fi
  if ! pgrep -f "train_swin_isic" > /dev/null; then
    echo "[watcher] swin process exited without COMPLETE flag, launching U-Net anyway"
    setsid bash /root/run_unet_isic.sh > /root/unet_isic18_run.log 2>&1 &
    break
  fi
  sleep 60
done
echo "[watcher] U-Net launched"
