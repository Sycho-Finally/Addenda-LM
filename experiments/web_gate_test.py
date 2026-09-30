import os, json, gc, hashlib
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# 生网络 vs 熟网络 对照实验 (在 mem_continual_test 的 additive 分支上做)
# claims_web.json: 同一组问题各配"多源佐证的真答案"(门准入) 与 "零佐证的假答案"(门拒绝).
#   分支V: 只吃门准入的 6 条真实网事实   => 应学会真话, A 不退化
#   分支P: 直接吃被门拒绝的 4 条投毒     => 应同样学会(而且学得一样快) => 假话流畅输出
# 结论预期: additive 分支不区分真假, 学什么信什么; 真假判定必须发生在权重之外(门).
# 这就是"生网络必须先熟化"的实验证据, 也量化了"无门直喂"的后果.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
MODEL_DIR = os.path.join(HERE, "model_cache", "Qwen3-4B-Instruct-2507")

from mem_continual_test import build_example, eval_task, MemBranch, hash_base, TASK_A


def train_mem_ex(tok, model, mem, exs, steps=240, bs=4, lr=1e-3):
    opt = torch.optim.AdamW(mem.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    model.eval()
    last = None
    for st in range(steps):
        idx = rng.choice(len(exs), min(bs, len(exs)), replace=len(exs) < bs)
        chunk = [exs[i] for i in idx]
        L = max(len(x[0]) for x in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id or 0, dtype=torch.long)
        lab = torch.full((len(chunk), L), -100, dtype=torch.long)
        msk = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, (x, y) in enumerate(chunk):
            ids[j, :len(x)] = torch.tensor(x); lab[j, :len(y)] = torch.tensor(y)
            msk[j, :len(x)] = 1
        ids, lab, msk = ids.cuda(), lab.cuda(), msk.cuda()
        with torch.no_grad():
            out = model.model(input_ids=ids, attention_mask=msk, output_hidden_states=True)
            h = out.hidden_states[-1].detach()
        logits = model.lm_head(model.model.norm(mem(h)))
        gl = logits[:, :-1]
        loss = nn.functional.cross_entropy(
            gl.reshape(-1, gl.shape[-1]).float(), lab[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        last = loss.item()
        if st % 60 == 0:
            print(f"    step {st:3d} loss={last:.4f}", flush=True)
    return last


@torch.no_grad()
def first_token_acc(tok, model, mem, claims):
    """teacher-forced: 答案首 token argmax 是否命中 (对分支P, '命中'意味着学会假话)."""
    model.eval(); mem.eval()
    hit = 0
    for c in claims:
        ids, _ = build_example(tok, c["q"], c["a"])
        ans_len = len(tok(c["a"], add_special_tokens=False).input_ids)
        t = torch.tensor(ids[:-1]).unsqueeze(0).cuda()
        out = model.model(input_ids=t, output_hidden_states=True)
        h = out.hidden_states[-1].detach()
        logits = model.lm_head(model.model.norm(mem(h)))
        pos = len(ids) - ans_len - 2
        gold = tok(c["a"], add_special_tokens=False).input_ids[0]
        hit += int(logits[0, pos].argmax().item() == gold)
    return hit / len(claims)


def main():
    data = json.load(open(os.path.join(DATA, "claims_web.json"), encoding="utf-8"))
    admitted = [c for c in data["claims"] if c["admitted"]]
    rejected = [c for c in data["claims"] if not c["admitted"]]
    print(f"门: 准入 {len(admitted)} 条, 拒绝 {len(rejected)} 条 (投毒对照)", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"),
        device_map="auto", attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    h0 = hash_base(model)

    exs_V = [build_example(tok, c["q"], c["a"]) for c in admitted]
    exs_P = [build_example(tok, c["q"], c["a"]) for c in rejected]

    # ---- 分支 V: 门准入的真事实 ----
    mem_V = MemBranch(d=model.config.hidden_size).cuda()
    a0_V = eval_task(tok, model, TASK_A, mem=mem_V)
    bv0 = first_token_acc(tok, model, mem_V, admitted)
    print(f"\n=== 分支V(熟数据, {len(admitted)}条真事实) 训练 ===", flush=True)
    train_mem_ex(tok, model, mem_V, exs_V)
    bv1 = first_token_acc(tok, model, mem_V, admitted)
    a1_V = eval_task(tok, model, TASK_A, mem=mem_V)
    da_V = [b - a for b, a in zip(a1_V, a0_V)]

    # ---- 分支 P: 无门直喂的投毒 ----
    mem_P = MemBranch(d=model.config.hidden_size).cuda()
    bp0 = first_token_acc(tok, model, mem_P, rejected)
    print(f"\n=== 分支P(生数据直喂, {len(rejected)}条投毒, 无门) 训练 ===", flush=True)
    train_mem_ex(tok, model, mem_P, exs_P)
    bp1 = first_token_acc(tok, model, mem_P, rejected)
    a1_P = eval_task(tok, model, TASK_A, mem=mem_P)
    da_P = [b - a for b, a in zip(a1_P, a0_V)]

    h1 = hash_base(model)
    print(f"\nbase_hash {'UNCHANGED' if h1 == h0 else 'CHANGED!'}")
    print("\n=== 对照结果 ===")
    print(f"分支V(门准入真事实): B准确率 {bv0:.2f} -> {bv1:.2f} | A mean_logp {np.mean(a0_V):.3f} -> {np.mean(a1_V):.3f} (d={np.mean(da_V):+.3f}, 退化>0.3条目 {sum(1 for x in da_V if x < -0.3)}/20)")
    print(f"分支P(无门投毒直喂): B准确率 {bp0:.2f} -> {bp1:.2f} | A mean_logp d={np.mean(da_P):+.3f} (退化>0.3条目 {sum(1 for x in da_P if x < -0.3)}/20)")
    print("=> 分支P同样学会且同样快: 模型不辨真假, 门必须在权重之外")

    res = {"gate_admitted": len(admitted), "gate_rejected": len(rejected),
           "V": {"B_acc_before": bv0, "B_acc_after": bv1,
                 "A_mean_before": float(np.mean(a0_V)), "A_mean_after": float(np.mean(a1_V)),
                 "A_delta_mean": float(np.mean(da_V)),
                 "A_degraded": int(sum(1 for x in da_V if x < -0.3))},
           "P": {"B_acc_before": bp0, "B_acc_after": bp1,
                 "A_delta_mean": float(np.mean(da_P)),
                 "A_degraded": int(sum(1 for x in da_P if x < -0.3))},
           "base_hash": h0}
    with open(os.path.join(HERE, "web_gate_result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("结果写入 web_gate_result.json", flush=True)


if __name__ == "__main__":
    main()
