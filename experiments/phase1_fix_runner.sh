#!/bin/bash
# 阶段一补跑: E3 LoRA (修正评测后) + E5 双技能 (修正键型后)
cd "C:/Users/30312/Desktop/仓库"
PY="C:/Users/30312/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
export HF_HUB_OFFLINE=1
echo "=== E3 LoRA baseline (fixed eval) ==="
$PY experiments/lora_baseline_test.py
echo "=== E5 dual skill (fixed keys) ==="
$PY experiments/dual_skill_test.py
echo "PHASE1_FIX_DONE"
