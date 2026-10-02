#!/bin/bash
# GPU 接力看门: 等待显存释放后自动启动 phase1 运行器
cd "C:/Users/30312/Desktop/仓库"
PY="C:/Users/30312/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
export HF_HUB_OFFLINE=1

echo "[watcher] 等待 GPU 释放 (每 60s 检查一次, 阈值 <1000MiB)..."
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "$used" ] && [ "$used" -lt 1000 ] 2>/dev/null; then
    echo "[watcher] GPU 已释放 (used=${used}MiB), 启动 phase1..."
    break
  fi
  echo "[watcher] GPU 仍占用 (${used}MiB), 60s 后再查"
  sleep 60
done

bash experiments/phase1_runner.sh
echo "[watcher] PHASE1_ALL_DONE"
