#!/bin/bash
echo "[queue] waiting for Swin scratch to finish..."
while true; do
  if grep -q "SWIN-UNET TRAINING COMPLETE" /root/swin_unet_scratch_d2262_run.log 2>/dev/null; then break; fi
  if ! pgrep -f "train_swin_isic" > /dev/null && [ -s /root/swin_unet_scratch_d2262_run.log ]; then break; fi
  sleep 300
done
echo "[queue] Swin done, launching PTU d2"
cd /root/pure_transformer_unet
setsid python3 train.py --depth3 2 --lr 1e-4 --tag d2 > /root/ptu_d2_run2.log 2>&1 &
D2=$!
while kill -0 $D2 2>/dev/null; do sleep 300; done
echo "[queue] PTU d2 done, launching PTU d6"
setsid python3 train.py --depth3 6 --lr 1e-4 --tag d6 > /root/ptu_d6_run2.log 2>&1 &
D6=$!
while kill -0 $D6 2>/dev/null; do sleep 300; done
echo "[queue] PTU d6 done, launching PTU-SYM12 (prep+train)"
chmod +x /root/run_ptu_sym.sh
setsid bash /root/run_ptu_sym.sh > /root/ptu_sym_prep.log 2>&1 &
echo "[queue] PTU-SYM12 launched, all queued jobs dispatched"
