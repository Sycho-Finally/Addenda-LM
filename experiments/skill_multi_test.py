import os, json
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# 技能注入 v4: 跨层联合分支 (模块在 L={6,12,18,24,30} 各拥有一支, 跨深度组合电路)
# v3 结论: 单点中层(L18)修复记忆(seen 1.000)但 unseen 仍 0.000.
# 假设: 程序 = 跨深度电路, 单点注入表达不出来; 多层联合训练让梯度在各深度分配角色
#       (浅层提数字身份 -> 中层查表 -> 深层承诺 token), 即 PKM/LoRA 的残差加法近亲.
# 控制: 同密钥/同16训练数字(seed 11)/同500步/CE-only; 指标加 unseen 逐位准确率.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model_cache", "Qwen3-4B-Instruct-2507")
LAYER_SET = [6, 12, 18, 24, 30]
N_TRAIN = int(os.environ.get("N_TRAIN", "16"))
STEPS = int(os.environ.get("STEPS", "500"))
TAG = os.environ.get("TAG", f"n{N_TRAIN}s{STEPS}")
SEED_DATA = int(os.environ.get("SEED_DATA", "11"))

from mem_continual_test import build_example, MemBranch, hash_base, TASK_A


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


class MultiHook:
    """active=True 时所有注册层注入各自分支; False 时全部直通."""
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
def gen_answer(tok, model, mh, q, max_new=12):
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


@torch.no_grad()
def measure(tok, model, mh, numbers):
    return [int(gen_answer(tok, model, mh, Q(n)) == enc(n)) for n in numbers]


@torch.no_grad()
def per_position_acc(tok, model, mh, numbers):
    """unseen 逐位准确率: 答案第 j 个字母 argmax 命中率 (比 exact match 灵敏)."""
    hits, tot = 0, 0
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
            pred = int(logits[st_ + j].argmax().item())
            gold = int(lab[st_ + j].item())
            if gold != tok.convert_tokens_to_ids("<|im_end|>"):
                hits += int(pred == gold); tot += 1
    mh.active = False
    return hits / max(1, tot)


def main():
    rng = np.random.default_rng(SEED_DATA)   # 数据种子可参数化(测试集固定不受影响)
    test_ns = gen_numbers(np.random.default_rng(999), 16)   # 固定测试集: 曲线各点共用
    train_ns = [n for n in gen_numbers(rng, N_TRAIN) if n not in set(test_ns)]
    assert not (set(train_ns) & set(test_ns))
    print(f"train={len(train_ns)} (过滤碰撞后)  test=16", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"),
        device_map="auto", attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    branches = {L: MemBranch(d=model.config.hidden_size).cuda() for L in LAYER_SET}
    mh = MultiHook(branches)
    handles = [model.model.layers[L].register_forward_hook(mh.make(L)) for L in LAYER_SET]
    n_par = sum(p.numel() for b in branches.values() for p in b.parameters())
    print(f"跨层分支: layers={LAYER_SET}  总参数={n_par}", flush=True)

    base_seen = measure(tok, model, mh, train_ns)
    base_unseen = measure(tok, model, mh, test_ns)
    print(f"base: seen={np.mean(base_seen):.3f} unseen={np.mean(base_unseen):.3f} (应≈0)", flush=True)

    exs = [build_example(tok, Q(n), enc(n)) for n in train_ns]
    print("=== 训练跨层联合分支 (%d 条, %d 步 CE) ===" % (N_TRAIN, STEPS), flush=True)
    opt = torch.optim.AdamW([p for b in branches.values() for p in b.parameters()], lr=1e-3)
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
        mh.active = False
        if st % 50 == 0:
            print(f"    step {st:3d} loss={loss.item():.4f}", flush=True)

    tr_seen = measure(tok, model, mh, train_ns)
    tr_unseen = measure(tok, model, mh, test_ns)
    pp_unseen = per_position_acc(tok, model, mh, test_ns)
    d_unseen = np.mean(tr_unseen) - np.mean(base_unseen)
    print(f"\ntrained: seen={np.mean(tr_seen):.3f}  unseen(exact)={np.mean(tr_unseen):.3f}  unseen逐位={pp_unseen:.3f}")
    print(f"Δunseen(exact) = {d_unseen:+.3f}   (对照: 单点L18 seen=1.000 unseen=0.000)")

    # A 漂移 (全部分支全局激活)
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
    print(f"A漂移(全分支全局): d={np.mean(dg):+.3f} 退化>0.3 {sum(1 for x in dg if x < -0.3)}/20")
    print(f"base_hash {'UNCHANGED' if h1 == '28ab9b2cebc0a035' else h1}")

    verdict = ("CIRCUIT-INDUCED" if d_unseen > 0.25
               else "PARTIAL-CIRCUIT" if d_unseen > 0.05 or pp_unseen > 0.35
               else "BOUNDARY-DEEPER-THAN-DEPTH")
    print(f"verdict: {verdict}")
    res = {"layers": LAYER_SET, "params": n_par,
           "base_seen": float(np.mean(base_seen)), "base_unseen": float(np.mean(base_unseen)),
           "trained_seen": float(np.mean(tr_seen)), "trained_unseen_exact": float(np.mean(tr_unseen)),
           "trained_unseen_perpos": float(pp_unseen),
           "delta_unseen": float(d_unseen),
           "A_drift_global": float(np.mean(dg)), "A_degraded": int(sum(1 for x in dg if x < -0.3)),
           "verdict": verdict,
           "single_mid_ref": {"seen": 1.000, "unseen": 0.0}}
    with open(os.path.join(HERE, f"skill_multi_result_{TAG}.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("结果写入 skill_multi_result.json", flush=True)


if __name__ == "__main__":
    main()
