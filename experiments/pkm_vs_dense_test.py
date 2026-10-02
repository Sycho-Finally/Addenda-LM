import os, json
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# R3: PKM 形态实验 — 模块容量从稠密 MLP 升级为 product-key 稀疏查找
# (对齐 Memory Layers at Scale 路线的最小复现).
# 密码任务本质是查表: digit→letter 的 10 条映射. 假设: 稀疏键值查找比稠密 MLP
# 更契合查表型映射, 同数据同预算下 unseen 泛化更高.
# PKMBranch: q=Wq·x → 折半与两组子键各取 top-8 → 64 个键值对 softmax 聚合
#            → W_out 回残差流. w_out 零初始化(起点=恒等), 键值可训练.
# 对照: 同数据(seed 11 / 64 训练 / 16 固定测试) 同预算(800 步) 的稠密跨层分支
#       (results/skill_multi_result_n64s800.json: unseen 0.375/逐位 0.722, A +2.419 6/20).
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model_cache", "Qwen3-4B-Instruct-2507")
LAYER_SET = [6, 12, 18, 24, 30]
N_KEYS = 32          # 每组子键数 (全键空间 32×32=1024 槽)
TOPK = 8             # 每半部分取 top-8 → 64 个候选对
R = 256              # 查询/值瓶颈维度

from mem_continual_test import build_example, hash_base, TASK_A
import pegp_multilayer_test as P   # gen_numbers / Q / enc / KEY


class PKMBranch(nn.Module):
    """product-key 记忆分支: 零初始化输出 → 起点恒等; 只有被检索的槽位参与计算."""
    def __init__(self, d, n_keys=N_KEYS, r=R, topk=TOPK):
        super().__init__()
        half = r // 2
        self.half, self.n_keys, self.topk = half, n_keys, topk
        self.wq = nn.Linear(d, r, bias=False, dtype=torch.float32)
        self.k1 = nn.Parameter(torch.randn(n_keys, half) * 0.02)
        self.k2 = nn.Parameter(torch.randn(n_keys, half) * 0.02)
        self.values = nn.Parameter(torch.randn(n_keys * n_keys, r) * 0.02)  # 随机初始化: 零值+零w_out 会双重死锁
        self.w_out = nn.Linear(r, d, bias=False, dtype=torch.float32)
        nn.init.normal_(self.wq.weight, 0, 0.02)
        nn.init.zeros_(self.w_out.weight)

    def forward(self, x):
        B, T, d = x.shape
        xf = x.float().reshape(-1, d)
        q = self.wq(xf)                                        # (N, r)
        q1, q2 = q[:, :self.half], q[:, self.half:]
        s1 = q1 @ self.k1.T                                    # (N, n_keys)
        s2 = q2 @ self.k2.T
        i1 = s1.topk(self.topk, dim=-1).indices
        i2 = s2.topk(self.topk, dim=-1).indices
        pair = s1.gather(1, i1).unsqueeze(2) + s2.gather(1, i2).unsqueeze(1)
        w = torch.softmax(pair.reshape(B * T, -1), dim=-1)     # (N, topk²)
        idx = (i1.unsqueeze(2) * self.n_keys + i2.unsqueeze(1)).reshape(B * T, -1)
        v = self.values[idx]                                   # (N, topk², r)
        out = (w.unsqueeze(-1) * v).sum(1)                     # (N, r)
        return x + self.w_out(out).to(x.dtype).reshape(B, T, d)


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
    acc = [int(gen_answer_keepactive(tok, model, mh, P.Q(n)) == P.enc(n)) for n in numbers]
    mh.active = False
    return acc


@torch.no_grad()
def per_position_acc(tok, model, mh, numbers):
    hits = tot = 0
    mh.active = True
    for n in numbers:
        ids, _ = build_example(tok, P.Q(n), P.enc(n))
        t = torch.tensor(ids[:-1]).unsqueeze(0).cuda()
        lab = torch.tensor(ids[1:]).cuda()
        logits = model(input_ids=t).logits[0]
        ans_len = len(tok(P.enc(n), add_special_tokens=False).input_ids)
        st_ = len(ids) - ans_len - 1
        for j in range(ans_len):
            gold = int(lab[st_ + j].item())
            if gold != tok.convert_tokens_to_ids("<|im_end|>"):
                hits += int(int(logits[st_ + j].argmax().item()) == gold); tot += 1
    mh.active = False
    return hits / max(1, tot)


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


def main():
    rng = np.random.default_rng(11)          # 与稠密分支对照完全同数据
    test_ns = P.gen_numbers(np.random.default_rng(999), 16)
    train_ns = [n for n in P.gen_numbers(rng, 64) if n not in set(test_ns)]

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"),
        device_map="auto", attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    branches = {L: PKMBranch(model.config.hidden_size).cuda() for L in LAYER_SET}
    mh = MultiHook(branches)
    handles = [model.model.layers[L].register_forward_hook(mh.make(L)) for L in LAYER_SET]
    n_par = sum(p.numel() for b in branches.values() for p in b.parameters())
    print(f"PKM 跨层分支: layers={LAYER_SET} n_keys={N_KEYS} topk={TOPK} r={R} 参数={n_par}", flush=True)

    base_unseen = measure(tok, model, mh, test_ns)
    print(f"base unseen={np.mean(base_unseen):.3f} (应≈0)", flush=True)

    exs = [build_example(tok, P.Q(n), P.enc(n)) for n in train_ns]
    print(f"=== 训练 PKM 分支 (64 条, 800 步, lr 1e-3) ===", flush=True)
    opt = torch.optim.AdamW([p for b in branches.values() for p in b.parameters()], lr=1e-3)
    rng2 = np.random.default_rng(0)
    for st in range(800):
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
        mh.active = False
        if st % 100 == 0:
            print(f"    step {st:3d} loss={loss.item():.4f}", flush=True)

    tr_seen = measure(tok, model, mh, train_ns)
    tr_unseen = measure(tok, model, mh, test_ns)
    pp = per_position_acc(tok, model, mh, test_ns)
    print(f"\ntrained(PKM): seen={np.mean(tr_seen):.3f}  unseen(exact)={np.mean(tr_unseen):.3f}  逐位={pp:.3f}")
    print(f"(对照: 同数据同预算稠密分支 unseen 0.375 / 逐位 0.722, 参数 6.5M)")

    # A 漂移 (全局激活)
    a_base = task_a_logp(tok, model, mh, False)
    a_pkm = task_a_logp(tok, model, mh, True)
    dg = [b - a for b, a in zip(a_pkm, a_base)]
    h1 = hash_base(model)
    print(f"A漂移(PKM全局): d={np.mean(dg):+.4f} 退化>0.3 {sum(1 for x in dg if x < -0.3)}/20")
    print(f"base_hash {'UNCHANGED' if h1 == '28ab9b2cebc0a035' else h1}")

    res = {"form": "PKM product-key", "layers": LAYER_SET, "n_keys": N_KEYS, "topk": TOPK,
           "r": R, "params": n_par, "n_train": 64, "steps": 800,
           "trained_seen": float(np.mean(tr_seen)),
           "trained_unseen_exact": float(np.mean(tr_unseen)),
           "trained_unseen_perpos": float(pp),
           "A_drift_mean": float(np.mean(dg)), "A_degraded": int(sum(1 for x in dg if x < -0.3)),
           "ref_dense_branch": {"unseen_exact": 0.375, "unseen_perpos": 0.722,
                                 "params": 6567680, "A_drift": 2.419, "A_degraded": 6},
           "base_hash_unchanged": h1 == "28ab9b2cebc0a035"}
    with open(os.path.join(HERE, "pkm_vs_dense_result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("结果写入 pkm_vs_dense_result.json", flush=True)


if __name__ == "__main__":
    main()
