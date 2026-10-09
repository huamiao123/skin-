#!/bin/bash
while true; do
  if grep -q "EGE-DUAL SCAUG COMPLETE" /root/experiments/logs/ege_dual_scaug_run.log 2>/dev/null && ! grep -q "SCAUG DONE" /root/experiments/logs/campaign_status.log 2>/dev/null; then
    echo "[$(date +%H:%M)] E-A SCAUG DONE" >> /root/experiments/logs/campaign_status.log
    grep -E "test of best" /root/experiments/logs/ege_dual_scaug_run.log | tail -1 >> /root/experiments/logs/campaign_status.log
  fi
  if grep -q "EGE-DUAL SCF COMPLETE" /root/experiments/logs/ege_dual_scf_run.log 2>/dev/null && ! grep -q "SCF DONE" /root/experiments/logs/campaign_status.log 2>/dev/null; then
    echo "[$(date +%H:%M)] E-B SCF DONE" >> /root/experiments/logs/campaign_status.log
    grep -E "test of best" /root/experiments/logs/ege_dual_scf_run.log | tail -1 >> /root/experiments/logs/campaign_status.log
    break
  fi
  sleep 300
done
