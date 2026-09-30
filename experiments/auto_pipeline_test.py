import os, json
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# 端到端闭环: raw_claims.json -> curator.py 规则门 -> 训练(replay锚定) -> 评测
# 验证"自动熟化管线"全链路: 门产出直接消费, 无人工核证环节.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
MODEL_DIR = os.path.join(HERE, "model_cache", "Qwen3-4B-Instruct-2507")

from mem_continual_test import build_example, eval_task, MemBranch, hash_base, TASK_A
from web_gate_fix import train_replay, first_token_acc


def main():
    cur = json.load(open(os.path.join(DATA, "curated_claims.json"), encoding="utf-8"))
    admitted = cur["admitted"]
    print(f"消费自动门产出: {len(admitted)} 条准入", flush=True)
    for c in admitted:
        print(f"  {c['a']:<14} <- {c['q'][:26]}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"),
        device_map="auto", attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    exs_V = [build_example(tok, c["q"], c["a"]) for c in admitted]
    exs_anchor = [build_example(tok, q, a) for q, a in TASK_A]

    mem = MemBranch(d=model.config.hidden_size).cuda()
    a0 = eval_task(tok, model, TASK_A, mem=mem)
    b0 = first_token_acc(tok, model, mem, admitted)
    print("\n=== 训练(自动门准入集 + replay锚定) ===", flush=True)
    train_replay(tok, model, mem, exs_V, exs_anchor)
    b1 = first_token_acc(tok, model, mem, admitted)
    a1 = eval_task(tok, model, TASK_A, mem=mem)
    da = [b - a for b, a in zip(a1, a0)]
    h1 = hash_base(model)
    print(f"\nB准确率 {b0:.2f} -> {b1:.2f}")
    print(f"A mean_logp {np.mean(a0):.3f} -> {np.mean(a1):.3f} (d={np.mean(da):+.3f}, 退化>0.3条目 {sum(1 for x in da if x < -0.3)}/20)")
    print(f"base_hash {'UNCHANGED' if h1 == '28ab9b2cebc0a035' else h1}")
    res = {"pipeline": "raw_web -> rule_gate -> anchored_training",
           "n_admitted": len(admitted),
           "B_acc_before": b0, "B_acc_after": b1,
           "A_delta_mean": float(np.mean(da)),
           "A_degraded": int(sum(1 for x in da if x < -0.3)),
           "base_hash_unchanged": h1 == "28ab9b2cebc0a035"}
    with open(os.path.join(HERE, "auto_pipeline_result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("结果写入 auto_pipeline_result.json", flush=True)


if __name__ == "__main__":
    main()
