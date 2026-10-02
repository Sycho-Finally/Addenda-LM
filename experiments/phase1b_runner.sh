#!/bin/bash
cd "C:/Users/30312/Desktop/仓库"
PY="C:/Users/30312/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
export HF_HUB_OFFLINE=1
echo "=== E2a SmolLM2-1.7B 不遗忘 (跨厂商) ==="
ADDENDA_MODEL_DIR="C:/Users/30312/Desktop/仓库/experiments/model_cache/SmolLM2-1.7B-Instruct" ADDENDA_TAG=_smollm2 $PY experiments/mem_continual_test.py
echo "=== E2b Qwen2.5-3B 路由+不遗忘 (跨代) ==="
ADDENDA_MODEL_DIR="C:/Users/30312/Desktop/仓库/experiments/model_cache/Qwen2.5-3B-Instruct" ADDENDA_TAG=_qwen25 $PY experiments/routing_test.py
echo "=== E4 真实知识注入全链路 (Qwen3-4B) ==="
$PY experiments/e4_real_injection_test.py
echo "PHASE1B_ALL_DONE"
