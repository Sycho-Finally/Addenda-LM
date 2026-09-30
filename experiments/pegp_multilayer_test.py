import os, json
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# 全局激活优化: PEGP 式零空间投影 (arXiv 2405.13383) 实例化到多层残差分支
# PEGP 抗遗忘条件: 旧任务输入特征 x_t 满足 x_t^T ΔE = 0 => 旧任务输出不变.
# 实例化到 mem(x) = W2·gelu(W1·x):
#   C1: ΔW1 · x_anchor = 0  (输入侧零空间)  => W1·x_anchor 恒定 => a_anchor=gelu(...) 恒定
#   C2: ΔW2 · a_anchor = 0  (输出侧零空间)  => 分支在锚输入上的输出恒 = 0 (W2 零初始化)
#   => 全局激活下, 旧世界函数逐位保持 (结构保证, 非 soft 正则).
#   bias 冻结 (b1=b2=0); r=1024 给 W2 留自由维度 (锚跨度 ~800 < 1024).
#   每步 opt.step() 后对权重再投影 => 约束对任意优化器成立.
# 对照: 无约束多层分支 (N=64): unseen 0.375/逐位0.722, A漂移 +2.419 (6/20).
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model_cache", "Qwen3-4B-Instruct-2507")
LAYER_SET = [6, 12, 18, 24, 30]
R = 1024
N_TRAIN = 64
STEPS = 800

from mem_continual_test import build_example, hash_base, TASK_A


def gen_numbers(rng, n, digits=5):
    out = set()
    while len(out) < n:
        ds = rng.choice(10, digits, replace=False)
        if ds[0] == 0:
            continue
        out.add("".join(str(d) for d in ds))
    return sorted(out)


KEY_SRC = "0123456789"
KEY_DST = "QMZXVKGBHD"
KEY = {int(d): c for d, c in zip(KEY_SRC, KEY_DST)}


def Q(n):
    return f"用内部密码表把数字 {n} 编码，只输出编码结果。"


def enc(n):
    return "".join(KEY[int(d)] for d in n)


class PBranch(nn.Module):
    """无 bias 版本 (bias 会破坏锚点保持, 冻结为零). float32 参数."""
    def __init__(self, d, r):
        super().__init__()
        self.w1 = nn.Linear(d, r, bias=False, dtype=torch.float32)
        self.w2 = nn.Linear(r, d, bias=False, dtype=torch.float32)
        nn.init.normal_(self.w1.weight, 0, 0.02)
        nn.init.zeros_(self.w2.weight)          # 零初始化 => 起点恒等

    def forward(self, x):                        # x bf16 (B,T,d)
        return x + self.w2(nn.functional.gelu(self.w1(x.float()))).to(x.dtype)


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
def gen_answer_keepactive(tok, model, mh, q, max_new=12):
    prompt_text = tok.apply_chat_template([{"role": "user", "content": q}],
                                          add_generation_prompt=True, tokenize=False)
    ids = tok(prompt_text, add_special_tokens=False).input_ids
    imend = tok.convert_tokens_to_ids("<|im_end|>")
    gen = []
    mh.active = True
    for _ in range(max_new):
        t = torch.tensor(ids).unsqueeze(0).cuda()
        out = model.model(input_ids=t).last_hidden_state
        logits = model.lm_head(out)
        nxt = int(logits[0, -1].argmax().item())
        if nxt == imend:
            break
        gen.append(nxt); ids.append(nxt)
    mh.active = False
    return tok.decode(gen).strip()


def measure(tok, model, mh, numbers):
    mh.active = True
    acc = [int(gen_answer_keepactive(tok, model, mh, Q(n)) == enc(n)) for n in numbers]
    mh.active = False
    return acc


@torch.no_grad()
def per_position_acc(tok, model, mh, numbers):
    hits = tot = 0
    mh.active = True
    for n in numbers:
        ids, _ = build_example(tok, Q(n), enc(n))
        t = torch.tensor(ids[:-1]).unsqueeze(0).cuda()
        lab = torch.tensor(ids[1:]).cuda()
        out = model.model(input_ids=t).last_hidden_state
        logits = model.lm_head(out)[0]
        ans_len = len(tok(enc(n), add_special_tokens=False).input_ids)
        st_ = len(ids) - ans_len - 1
        for j in range(ans_len):
            gold = int(lab[st_ + j].item())
            if gold != tok.convert_tokens_to_ids("<|im_end|>"):
                hits += int(int(logits[st_ + j].argmax().item()) == gold); tot += 1
    mh.active = False
    return hits / max(1, tot)


@torch.no_grad()
def collect_anchor_X(model, tok, layer_set, a_exs):
    """钩子直通状态下, 收集各注入层的基座残差.
    注入点在 layers[L] 的输出 = hidden_states[L+1] (hidden_states[L] 是该层输入, 差一层!)."""
    Xs = {L: [] for L in layer_set}
    for ids, _ in a_exs:
        t = torch.tensor(ids).unsqueeze(0).cuda()
        out = model.model(input_ids=t, output_hidden_states=True)
        for L in layer_set:
            Xs[L].append(out.hidden_states[L + 1][0].float().cpu())
    return {L: torch.cat(v, 0).T.contiguous() for L, v in Xs.items()}   # (d, M)


@torch.no_grad()
def a_acts(branch, X):
    """a = gelu(W1 x)  (r, M)"""
    return torch.nn.functional.gelu(branch.w1.weight @ X)


def main():
    rng = np.random.default_rng(11)
    test_ns = gen_numbers(np.random.default_rng(999), 16)
    train_ns = [n for n in gen_numbers(rng, N_TRAIN) if n not in set(test_ns)]
    print(f"train={len(train_ns)} test=16 (不相交)", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"),
        device_map="auto", attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    branches = {L: PBranch(model.config.hidden_size, R).cuda() for L in LAYER_SET}
    mh = MultiHook(branches)
    handles = [model.model.layers[L].register_forward_hook(mh.make(L)) for L in LAYER_SET]
    n_par = sum(p.numel() for b in branches.values() for p in b.parameters())
    print(f"PEGP 多层分支: layers={LAYER_SET} r={R} params={n_par}", flush=True)

    # ---- 锚特征 + 零空间基 ----
    a_exs = [build_example(tok, q, a) for q, a in TASK_A]
    print("收集锚特征 (钩子直通, 纯基座残差)...", flush=True)
    X_anchor = collect_anchor_X(model, tok, LAYER_SET, a_exs)
    w10, Ux, Ua = {}, {}, {}
    for L in LAYER_SET:
        X = X_anchor[L].cpu()                              # (d, M)
        U, S, Vh = torch.linalg.svd(X.T, full_matrices=False)
        Ux[L] = Vh.T.contiguous().cuda()                   # (d, k) 锚输入空间正交基
        w10[L] = branches[L].w1.weight.data.clone()
        aL = a_acts(branches[L], X_anchor[L].cuda()).cpu() # (r, M) 初始 a
        U, S, Vh = torch.linalg.svd(aL.T, full_matrices=False)
        Ua[L] = Vh.T.contiguous().cuda()                   # (r, k') 锚a空间正交基
        free_w1 = model.config.hidden_size - Ux[L].shape[1]
        free_w2 = R - Ua[L].shape[1]
        print(f"  L{L}: 锚位置={X.shape[1]}  W1自由维={free_w1}  W2自由维={free_w2}", flush=True)

    base_seen = measure(tok, model, mh, train_ns)
    base_unseen = measure(tok, model, mh, test_ns)
    print(f"base: seen={np.mean(base_seen):.3f} unseen={np.mean(base_unseen):.3f} (应≈0)", flush=True)

    exs = [build_example(tok, Q(n), enc(n)) for n in train_ns]
    print(f"=== 训练 (PEGP 投影, {len(train_ns)} 条, {STEPS} 步) ===", flush=True)
    opt = torch.optim.AdamW([p for b in branches.values() for p in b.parameters()],
                            lr=1e-3, weight_decay=0.0)
    rng2 = np.random.default_rng(0)
    for st in range(STEPS):
        idx = rng2.choice(len(exs), 4)
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
        # ---- PEGP 再投影 (优化器不可知的约束执行) ----
        with torch.no_grad():
            for L in LAYER_SET:
                br = branches[L]
                br.w1.weight.data -= (br.w1.weight.data - w10[L]) @ Ux[L] @ Ux[L].T
                # W2 零初始化 => W2^0 = 0, 约束即 W2 U_a U_a^T = 0
                br.w2.weight.data -= br.w2.weight.data @ Ua[L] @ Ua[L].T
        mh.active = False
        if st % 50 == 0:
            # 锚输出范数监控 (应≈0 => 旧世界保持)
            with torch.no_grad():
                def branch_delta(L):
                    xt = X_anchor[L].cuda().T
                    return (branches[L](xt) - xt).abs().mean()   # 只量分支增量, 不含残差本身
                nrm = float(torch.stack([branch_delta(L) for L in LAYER_SET]).mean())
            print(f"    step {st:3d} loss={loss.item():.4f} anchor|mem|={nrm:.2e}", flush=True)

    tr_seen = measure(tok, model, mh, train_ns)
    tr_unseen = measure(tok, model, mh, test_ns)
    pp = per_position_acc(tok, model, mh, test_ns)
    d_unseen = np.mean(tr_unseen) - np.mean(base_unseen)
    print(f"\ntrained(PEGP): seen={np.mean(tr_seen):.3f}  unseen(exact)={np.mean(tr_unseen):.3f}  逐位={pp:.3f}")
    print(f"Δunseen = {d_unseen:+.3f}   (对照: 无约束多层 0.375/0.722)")

    # A 漂移 (全局激活) — 预期 ~0 (结构保证)
    a_base_s, a_mid_s = [], []
    with torch.no_grad():
        for q, a in TASK_A:
            ids, _ = build_example(tok, q, a)
            t = torch.tensor(ids[:-1]).unsqueeze(0).cuda()
            lab = torch.tensor(ids[1:]).cuda()
            mh.active = False
            o0 = model.model(input_ids=t).last_hidden_state
            lp0 = torch.log_softmax(model.lm_head(o0)[0].float(), -1)
            mh.active = True
            o1 = model.model(input_ids=t).last_hidden_state
            lp1 = torch.log_softmax(model.lm_head(o1)[0].float(), -1)
            ans_len = len(tok(a, add_special_tokens=False).input_ids)
            st_ = len(ids) - ans_len - 1
            idx = torch.arange(st_, st_ + ans_len).cuda()
            a_base_s.append(float(lp0[idx, lab[idx]].mean()))
            a_mid_s.append(float(lp1[idx, lab[idx]].mean()))
    mh.active = False
    dg = [b - a for b, a in zip(a_mid_s, a_base_s)]
    h1 = hash_base(model)
    print(f"A漂移(全局激活+PEGP): d={np.mean(dg):+.6f}  max|d|={np.max(np.abs(dg)):.6f}  "
          f"退化>0.3 {sum(1 for x in dg if x < -0.3)}/20   (对照无投影: +2.419, 6/20)")
    print(f"base_hash {'UNCHANGED' if h1 == '28ab9b2cebc0a035' else h1}")

    verdict = ("PEGP-WORKS" if abs(np.mean(dg)) < 0.05 and d_unseen > 0.05
               else "PEGP-PRESERVES-BUT-NO-LEARN" if abs(np.mean(dg)) < 0.05
               else "CHECK")
    print(f"verdict: {verdict}")
    res = {"layers": LAYER_SET, "r": R, "params": n_par, "n_train": len(train_ns),
           "base_unseen": float(np.mean(base_unseen)),
           "trained_seen": float(np.mean(tr_seen)),
           "trained_unseen_exact": float(np.mean(tr_unseen)),
           "trained_unseen_perpos": float(pp),
           "delta_unseen": float(d_unseen),
           "A_drift_mean": float(np.mean(dg)), "A_drift_maxabs": float(np.max(np.abs(dg))),
           "A_degraded": int(sum(1 for x in dg if x < -0.3)),
           "ref_unconstrained": {"unseen_exact": 0.375, "unseen_perpos": 0.722,
                                  "A_drift": 2.419, "A_degraded": 6},
           "verdict": verdict}
    with open(os.path.join(HERE, "pegp_result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("结果写入 pegp_result.json", flush=True)


if __name__ == "__main__":
    main()
