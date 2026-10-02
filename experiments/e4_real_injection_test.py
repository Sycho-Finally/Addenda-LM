import os, json
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# E4: 真实知识注入全链路 — 门核证的真实网络事实(data/curated_claims.json,
# 均为基座 cutoff 之后的事件, 基座不可能知道) → 跨层分支注入 → 模型能答 + 旧任务不动.
# 指标: 每条事实 生成命中(宽松=答案子串 / 严格=完全一致) 训练前后对比
#       + TASK_A 漂移(全局激活) + 主干哈希.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model_cache", "Qwen3-4B-Instruct-2507")
LAYER_SET = [6, 12, 18, 24, 30]
R = 1024

from mem_continual_test import build_example, hash_base, TASK_A
import pegp_multilayer_test as P   # PBranch

with open(os.path.join(os.path.dirname(HERE), "data", "curated_claims.json"), encoding="utf-8") as f:
    claims = [c for c in json.load(f)["admitted"]]
FACTS = [(c["q"], c["a"]) for c in claims]


class MultiHook:
    """active=True 时各层注入各自分支; False 时全部直通(逐位=基座)."""
    def __init__(self, branches):
        self.branches = branches
        self.active = False

    def make(self, L):
        def hook(module, args, output):
            if not self.active:
                return output
            mem = self.branches[L]
            if isinstance(output, tuple):
                return (output[0] + mem(output[0]),) + output[1:]
            return output + mem(output)
        return hook


@torch.no_grad()
def gen_answer(tok, model, mh, q, max_new=14):
    prompt_text = tok.apply_chat_template([{"role": "user", "content": q}],
                                          add_generation_prompt=True, tokenize=False)
    ids = tok(prompt_text, add_special_tokens=False).input_ids
    imend = tok.convert_tokens_to_ids("<|im_end|>")
    gen = []
    mh.active = True
    for _ in range(max_new):
        t = torch.tensor(ids).unsqueeze(0).cuda()
        logits = model(input_ids=t).logits
        nxt = int(logits[0, -1].argmax().item())
        if nxt == imend:
            break
        gen.append(nxt); ids.append(nxt)
    mh.active = False
    return tok.decode(gen).strip()


@torch.no_grad()
def measure_facts(tok, model, mh):
    out = []
    mh.active = True
    for q, a in FACTS:
        pred = gen_answer_inner(tok, model, q)
        out.append((int(a in pred), int(pred.strip() == a), pred))
    mh.active = False
    return out


def gen_answer_inner(tok, model, q):
    prompt_text = tok.apply_chat_template([{"role": "user", "content": q}],
                                          add_generation_prompt=True, tokenize=False)
    ids = tok(prompt_text, add_special_tokens=False).input_ids
    imend = tok.convert_tokens_to_ids("<|im_end|>")
    gen = []
    for _ in range(14):
        t = torch.tensor(ids).unsqueeze(0).cuda()
        logits = model(input_ids=t).logits
        nxt = int(logits[0, -1].argmax().item())
        if nxt == imend:
            break
        gen.append(nxt); ids.append(nxt)
    return tok.decode(gen).strip()


@torch.no_grad()
def task_a_logp(tok, model, mh, active):
    scores = []
    mh.active = active
    for q, a in TASK_A:
        ids, _ = build_example(tok, q, a)
        t = torch.tensor(ids[:-1]).unsqueeze(0).cuda()
        lab = torch.tensor(ids[1:]).cuda()
        logp = torch.log_softmax(model(input_ids=t).logits[0].float(), -1)
        ans_len = len(tok(a, add_special_tokens=False).input_ids)
        st_ = len(ids) - ans_len - 1
        idx = torch.arange(st_, st_ + ans_len).cuda()
        scores.append(float(logp[idx, lab[idx]].mean()))
    mh.active = False
    return scores


def main():
    print(f"门核证事实 {len(FACTS)} 条:", flush=True)
    for q, a in FACTS:
        print(f"  {a:<14} <- {q[:32]}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"),
        device_map="auto", attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    branches = {L: P.PBranch(model.config.hidden_size, R).cuda() for L in LAYER_SET}
    mh = MultiHook(branches)
    handles = [model.model.layers[L].register_forward_hook(mh.make(L)) for L in LAYER_SET]

    # ---- 基线 (不注入 = 纯基座) ----
    base_facts = measure_facts(tok, model, mh)
    print(f"base: 宽松命中={np.mean([r[0] for r in base_facts]):.3f}  "
          f"完全一致={np.mean([r[1] for r in base_facts]):.3f}", flush=True)
    for (q, a), (hit, exact, pred) in zip(FACTS, base_facts):
        print(f"  [base] {a:<14} pred={pred[:24]!r:<28} hit={hit}", flush=True)
    a_base = task_a_logp(tok, model, mh, False)

    exs = [build_example(tok, q, a) for q, a in FACTS]
    print(f"\n=== 训练跨层分支 (真实事实 {len(FACTS)} 条, 800 步) ===", flush=True)
    opt = torch.optim.AdamW([p for L in LAYER_SET for p in branches[L].parameters()], lr=1e-3)
    rng2 = np.random.default_rng(0)
    for st in range(800):
        idx = rng2.choice(len(exs), min(4, len(exs)))
        chunk = [exs[i] for i in idx]
        L = max(len(x[0]) for x in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id or 0, dtype=torch.long)
        lab = torch.full((len(chunk), L), -100, dtype=torch.long)
        msk = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, (x, y) in enumerate(chunk):
            ids[j, :len(x)] = torch.tensor(x); lab[j, :len(y)] = torch.tensor(y)
            msk[j, :len(x)] = 1
        ids, lab, msk = ids.cuda(), lab.cuda(), msk.cuda()
        mh.active = True
        out = model.model(input_ids=ids, attention_mask=msk).last_hidden_state
        logits = model.lm_head(out)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            lab[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        mh.active = False
        if st % 100 == 0:
            print(f"    step {st:3d} loss={loss.item():.4f}", flush=True)

    tr_facts = measure_facts(tok, model, mh)
    print(f"\ntrained: 宽松命中={np.mean([r[0] for r in tr_facts]):.3f}  "
          f"完全一致={np.mean([r[1] for r in tr_facts]):.3f}")
    for (q, a), (hit, exact, pred) in zip(FACTS, tr_facts):
        print(f"  [trained] {a:<14} pred={pred[:24]!r:<28} hit={hit} exact={exact}", flush=True)

    a_tr = task_a_logp(tok, model, mh, True)
    dg = [b - a for b, a in zip(a_tr, a_base)]
    h1 = hash_base(model)
    print(f"\nA漂移(全局激活): mean={np.mean(dg):+.4f} 退化>0.3 {sum(1 for x in dg if x < -0.3)}/20")
    print(f"base_hash {'UNCHANGED' if h1 == '28ab9b2cebc0a035' else h1}")

    res = {"n_facts": len(FACTS),
           "base_hit_loose": float(np.mean([r[0] for r in base_facts])),
           "trained_hit_loose": float(np.mean([r[0] for r in tr_facts])),
           "base_exact": float(np.mean([r[1] for r in base_facts])),
           "trained_exact": float(np.mean([r[1] for r in tr_facts])),
           "A_drift_mean": float(np.mean(dg)), "A_degraded": int(sum(1 for x in dg if x < -0.3)),
           "base_hash_unchanged": h1 == "28ab9b2cebc0a035",
           "per_fact": [{"q": q, "gold": a,
                          "base_pred": bf[2], "trained_pred": tf[2],
                          "trained_loose_hit": tf[0], "trained_exact": tf[1]}
                         for (q, a), bf, tf in zip(FACTS, base_facts, tr_facts)]}
    with open(os.path.join(HERE, "e4_real_injection_result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("结果写入 e4_real_injection_result.json", flush=True)


if __name__ == "__main__":
    main()
