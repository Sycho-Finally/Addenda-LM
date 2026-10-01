#!/bin/bash
# 阶段一运行器: E1 多种子 / E3 LoRA 对照 / E5 双技能干扰 (顺序执行, 单卡串行)
cd "C:/Users/30312/Desktop/仓库"
PY="C:/Users/30312/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
export HF_HUB_OFFLINE=1

echo "=== E1 s101 n64 ==="
SEED_DATA=101 N_TRAIN=64 STEPS=800 TAG=e1_s101_n64 $PY experiments/skill_multi_test.py
echo "=== E1 s101 n128 ==="
SEED_DATA=101 N_TRAIN=128 STEPS=800 TAG=e1_s101_n128 $PY experiments/skill_multi_test.py
echo "=== E1 s102 n64 ==="
SEED_DATA=102 N_TRAIN=64 STEPS=800 TAG=e1_s102_n64 $PY experiments/skill_multi_test.py
echo "=== E1 s102 n128 ==="
SEED_DATA=102 N_TRAIN=128 STEPS=800 TAG=e1_s102_n128 $PY experiments/skill_multi_test.py
echo "=== E3 LoRA baseline ==="
$PY experiments/lora_baseline_test.py
echo "=== E5 dual skill ==="
$PY experiments/dual_skill_test.py
echo "PHASE1_ALL_DONE"
