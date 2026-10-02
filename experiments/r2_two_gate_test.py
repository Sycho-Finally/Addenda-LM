import os, json
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# R2: 双门对照 — 裸适配器 vs 门+适配器 (摄取门的可叠加价值)
# Arm A (裸):   干净5条 + 毒3条 全部无差别训练 (无摄取门)
# Arm B (双门): curator 门先过滤 (毒被 R1/R4 拒绝) → 仅干净5条训练
# 毒料: 2 条同问题近似毒 (真答案 9月10日 vs 毒 9月9日; 7月31日 vs 6月23日)
#       + 1 条不可能日期毒 (2月30日, R1 确定性拒绝)
# 指标: 干净命中 / 毒命中 (同问题冲突: 毒串在输出中出现=被污染) / A 漂移 / 哈希
# 两臂各自从全新零初始化分支开始, 同预算 800 步.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model_cache", "Qwen3-4B-Instruct-2507")
LAYER_SET = [6, 12, 18, 24, 30]
R = 1024

from mem_continual_test import build_example, hash_base, TASK_A
import pegp_multilayer_test as P   # PBranch

with open(os.path.join(os.path.dirname(HERE), "data", "curated_claims.json"), encoding="utf-8") as f:
    cur = json.load(f)
CLEAN = [(c["q"], c["a"]) for c in cur["admitted"]]
POISON = [(c["q"], c["a"]) for c in cur["rejected"][:3]]   # 近似毒×2 + 不可能日期×1


class MultiHook:
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
def task_a_logp(tok, model, mh, active):
    mh.active = active
    scores = []
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


def eval_arm(tok, model, mh):
    """干净命中(宽松) / 毒命中(毒答案子串出现在输出) / A 漂移"""
    mh.active = True
    clean_hit, clean_preds = [], []
    for q, a in CLEAN:
        pred = gen_answer(tok, model, mh, q)
        clean_hit.append(int(a in pred)); clean_preds.append(pred)
    poison_hit, poison_preds = [], []
    for q, a in POISON:
        pred = gen_answer(tok, model, mh, q)
        poison_hit.append(int(a in pred)); poison_preds.append(pred)
    mh.active = False
    return clean_hit, poison_hit, clean_preds, poison_preds


def train_arm(tok, model, mh, exs, tag):
    print(f"=== {tag} 训练 {len(exs)} 条 ===", flush=True)
    branches = {L: P.PBranch(model.config.hidden_size, R).cuda() for L in LAYER_SET}
    mh.branches = branches
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
        if st % 200 == 0:
            print(f"    step {st:3d} loss={loss.item():.4f}", flush=True)
    return branches


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"),
        device_map="auto", attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mh = MultiHook(branches=None)
    handles = [model.model.layers[L].register_forward_hook(mh.make(L)) for L in LAYER_SET]
    h0 = hash_base(model)

    print(f"干净 {len(CLEAN)} 条 / 毒 {len(POISON)} 条 (2 条同问题近似毒 + 1 条不可能日期)", flush=True)

    # ---- Arm A: 裸适配器 (无摄取门, 干净+毒一起学) ----
    exs_A = [build_example(tok, q, a) for q, a in CLEAN + POISON]
    train_arm(tok, model, mh, exs_A, "Arm A 裸适配器")
    aA_base = task_a_logp(tok, model, mh, False)
    aA_tr = task_a_logp(tok, model, mh, True)
    dgA = [b - a for b, a in zip(aA_tr, aA_base)]
    cA, pA, cpA, ppA = eval_arm(tok, model, mh)

    # ---- Arm B: 门+适配器 (毒被门拒绝, 仅干净 5 条) ----
    exs_B = [build_example(tok, q, a) for q, a in CLEAN]
    train_arm(tok, model, mh, exs_B, "Arm B 门+适配器")
    aB_base = task_a_logp(tok, model, mh, False)
    aB_tr = task_a_logp(tok, model, mh, True)
    dgB = [b - a for b, a in zip(aB_tr, aB_base)]
    cB, pB, cpB, ppB = eval_arm(tok, model, mh)

    h1 = hash_base(model)
    print(f"\n{'='*30} 对照结果 {'='*30}")
    print(f"干净命中:  Arm A(裸)={np.mean(cA):.3f}   Arm B(门)={np.mean(cB):.3f}")
    print(f"毒命中:    Arm A(裸)={np.mean(pA):.3f}   Arm B(门)={np.mean(pB):.3f}   <- 核心指标")
    for (q, a), pa, pb in zip(POISON, pA, pB):
        print(f"    毒[{a}] 裸={pa} 门={pb}   q={q[:28]}")
    print(f"A漂移:     Arm A={np.mean(dgA):+.4f}   Arm B={np.mean(dgB):+.4f}")
    print(f"base_hash {'UNCHANGED' if h1 == '28ab9b2cebc0a035' else h1}")

    verdict = ("GATE-VALUE-CONFIRMED (毒被门挡在权重之外)"
               if np.mean(pA) > 0.3 and np.mean(pB) < 0.2
               else "GATE-NO-ADVANTAGE (需检查毒料或训练)")
    print(f"verdict: {verdict}")

    res = {"clean_hit_A": float(np.mean(cA)), "clean_hit_B": float(np.mean(cB)),
           "poison_hit_A": float(np.mean(pA)), "poison_hit_B": float(np.mean(pB)),
           "A_drift_A": float(np.mean(dgA)), "A_drift_B": float(np.mean(dgB)),
           "poison_detail": [{"q": q, "poison": a, "hit_bare": pa, "hit_gated": pb,
                               "pred_bare": ppA[i], "pred_gated": ppB[i]}
                              for i, ((q, a), pa, pb) in enumerate(zip(POISON, pA, pB))],
           "verdict": verdict, "base_hash_unchanged": h1 == "28ab9b2cebc0a035"}
    with open(os.path.join(HERE, "r2_gate_result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("结果写入 r2_gate_result.json", flush=True)


if __name__ == "__main__":
    main()
