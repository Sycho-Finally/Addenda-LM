import os, json
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# 行为熟化 demo: 内容门(真伪)之外, 再加 replay 锚定(对任务A背景分布做 KL-to-base),
# 检验"真事实训练导致的 A 漂移(-7.455, 14/20)"能否被压回.
# 这是补丁四(consolidation/sleep replay)的最小实现, 也是"熟网络"的第二层: 行为熟化.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
MODEL_DIR = os.path.join(HERE, "model_cache", "Qwen3-4B-Instruct-2507")

from mem_continual_test import build_example, eval_task, MemBranch, hash_base, TASK_A


def batch_h(tok, model, questions, answers, bs):
    exs = [build_example(tok, q, a) for q, a in zip(questions, answers)]
    L = max(len(x[0]) for x in exs)
    ids = torch.full((len(exs), L), tok.pad_token_id or 0, dtype=torch.long)
    lab = torch.full((len(exs), L), -100, dtype=torch.long)
    msk = torch.zeros((len(exs), L), dtype=torch.long)
    for j, (x, y) in enumerate(exs):
        ids[j, :len(x)] = torch.tensor(x); lab[j, :len(y)] = torch.tensor(y)
        msk[j, :len(x)] = 1
    return ids.cuda(), lab.cuda(), msk.cuda()


def train_replay(tok, model, mem, exs_new, exs_anchor, steps=240, bs=4, lr=1e-3, lam=2.0):
    """loss = CE(新事实) + lam * KL(p_base || p_mem+branch) on 旧任务背景批"""
    opt = torch.optim.AdamW(mem.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    model.eval()
    for st in range(steps):
        idx = rng.choice(len(exs_new), bs)
        chunk = [exs_new[i] for i in idx]
        L = max(len(x[0]) for x in chunk)
        ids = torch.full((bs, L), tok.pad_token_id or 0, dtype=torch.long)
        lab = torch.full((bs, L), -100, dtype=torch.long)
        msk = torch.zeros((bs, L), dtype=torch.long)
        for j, (x, y) in enumerate(chunk):
            ids[j, :len(x)] = torch.tensor(x); lab[j, :len(y)] = torch.tensor(y)
            msk[j, :len(x)] = 1
        ids, lab, msk = ids.cuda(), lab.cuda(), msk.cuda()
        with torch.no_grad():
            out = model.model(input_ids=ids, attention_mask=msk, output_hidden_states=True)
            h = out.hidden_states[-1].detach()
        logits = model.lm_head(model.model.norm(mem(h)))
        ce = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            lab[:, 1:].reshape(-1), ignore_index=-100)
        # ---- replay 锚: 旧任务批, KL(基座分布 || 当前分布) ----
        aidx = rng.choice(len(exs_anchor), 2)
        a_chunk = [exs_anchor[i] for i in aidx]
        La = max(len(x[0]) for x in a_chunk)
        a_ids = torch.full((len(a_chunk), La), tok.pad_token_id or 0, dtype=torch.long)
        a_msk = torch.zeros((len(a_chunk), La), dtype=torch.long)
        for j, (x, y) in enumerate(a_chunk):
            a_ids[j, :len(x)] = torch.tensor(x); a_msk[j, :len(x)] = 1
        with torch.no_grad():
            outa = model.model(input_ids=a_ids.cuda(), attention_mask=a_msk.cuda(),
                               output_hidden_states=True)
            ha = outa.hidden_states[-1].detach()
            base_logits = model.lm_head(model.model.norm(ha))
            p_base = torch.softmax(base_logits.float(), -1)
        mem_logits = model.lm_head(model.model.norm(mem(ha)))
        logp_mem = torch.log_softmax(mem_logits.float(), -1)
        kl = (p_base * (torch.log(p_base + 1e-12) - logp_mem)).sum(-1)
        kl = kl[a_msk.cuda().bool()].mean() if a_msk.cuda().bool().any() else kl.mean()
        loss = ce + lam * kl
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 60 == 0:
            print(f"    step {st:3d} ce={ce.item():.4f} kl={kl.item():.4f}", flush=True)


@torch.no_grad()
def first_token_acc(tok, model, mem, claims):
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

    mem_R = MemBranch(d=model.config.hidden_size).cuda()
    a0 = eval_task(tok, model, TASK_A, mem=mem_R)
    b0 = first_token_acc(tok, model, mem_R, admitted)
    print("=== 分支VR: 真事实 CE + 任务A replay KL 锚定 ===", flush=True)
    train_replay(tok, model, mem_R, exs_V, exs_anchor)
    b1 = first_token_acc(tok, model, mem_R, admitted)
    a1 = eval_task(tok, model, TASK_A, mem=mem_R)
    da = [b - a for b, a in zip(a1, a0)]
    print(f"\nB准确率 {b0:.2f} -> {b1:.2f}")
    print(f"A mean_logp {np.mean(a0):.3f} -> {np.mean(a1):.3f} (d={np.mean(da):+.3f}, 退化>0.3条目 {sum(1 for x in da if x < -0.3)}/20)")
    print(f"(对照: 无锚定的分支V是 d=-7.455, 14/20)")
    res = {"B_acc_before": b0, "B_acc_after": b1,
           "A_mean_before": float(np.mean(a0)), "A_mean_after": float(np.mean(a1)),
           "A_delta_mean": float(np.mean(da)),
           "A_degraded": int(sum(1 for x in da if x < -0.3)),
           "baseline_unanchored": {"A_delta_mean": -7.455, "A_degraded": 14}}
    with open(os.path.join(HERE, "web_gate_fix_result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("结果写入 web_gate_fix_result.json", flush=True)


if __name__ == "__main__":
    main()
