import os, json
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# 技能注入 v3: 中层模块 (插入 layer 18 / 共36层) — 直接攻 v2 发现的边界
# v2 结论: 顶层 additive 分支 学密码编码 seen 0.812 / unseen 0.000 (MEMORIZATION-ONLY).
# 机理假设: 顶层拿到的末层隐状态不含可供查表的逐位数字身份; 中层注入则让下游 17 层
#           冻结层继续加工注入信号 —— 基座预训练电路成为模块函数的一部分.
# 对照: 同一密钥 / 同一批数字 / 同参数量(1.3M) / 同训练步数, 唯一变量 = 插入深度.
# 实现: forward hook 挂在 model.model.layers[18], 输出残差加 mem(x);
#       active=None 时直通(逐位等于基座); 梯度经 17 层冻结层回传到分支.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model_cache", "Qwen3-4B-Instruct-2507")
LAYER_IDX = 18

from mem_continual_test import (MemBranch, hash_base, TASK_A, build_example)


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


class Hook:
    """层输出残差注入: active=mem 时 x' = x + mem(x); active=None 时直通."""
    def __init__(self):
        self.active = None

    def __call__(self, module, args, output):
        mem = self.active
        if mem is None:
            return output
        if isinstance(output, tuple):
            hs = output[0] + mem(output[0])
            return (hs,) + output[1:]
        return output + mem(output)


def measure(tok, model, hook, mem, numbers):
    hook.active = mem
    acc = []
    for n in numbers:
        acc.append(int(gen_answer_keepactive(tok, model, hook, Q(n)) == enc(n)))
    hook.active = None
    return acc


@torch.no_grad()
def gen_answer_keepactive(tok, model, hook, q, max_new=12):
    prompt_text = tok.apply_chat_template([{"role": "user", "content": q}],
                                          add_generation_prompt=True, tokenize=False)
    ids = tok(prompt_text, add_special_tokens=False).input_ids
    imend = tok.convert_tokens_to_ids("<|im_end|>")
    gen = []
    for _ in range(max_new):
        t = torch.tensor(ids).unsqueeze(0).cuda()
        out = model.model(input_ids=t).last_hidden_state
        logits = model.lm_head(out)
        nxt = int(logits[0, -1].argmax().item())
        if nxt == imend:
            break
        gen.append(nxt); ids.append(nxt)
    return tok.decode(gen).strip()


def main():
    rng = np.random.default_rng(11)          # 与 v2 相同种子 => 相同数字集与密钥
    train_ns = gen_numbers(rng, 16)
    test_ns = gen_numbers(rng, 16)
    assert not (set(train_ns) & set(test_ns))

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"),
        device_map="auto", attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hook = Hook()
    mem_mid = MemBranch(d=model.config.hidden_size).cuda()
    h = model.model.layers[LAYER_IDX].register_forward_hook(hook)
    n_par = sum(p.numel() for p in mem_mid.parameters())
    print(f"中层模块插入 layers[{LAYER_IDX}] (共{len(model.model.layers)}层), params={n_par}", flush=True)

    base_seen = measure(tok, model, hook, None, train_ns)
    base_unseen = measure(tok, model, hook, None, test_ns)
    print(f"base: seen={np.mean(base_seen):.3f} unseen={np.mean(base_unseen):.3f} (应≈0)", flush=True)

    exs = [build_example_cipher(tok, Q(n), enc(n)) for n in train_ns]
    print("=== 训练中层模块 (16 条, 500 步 CE, 梯度穿17层冻结层) ===", flush=True)
    opt = torch.optim.AdamW(mem_mid.parameters(), lr=1e-3)
    rng2 = np.random.default_rng(0)
    hook.active = mem_mid
    for st in range(500):
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
        out = model.model(input_ids=ids, attention_mask=msk).last_hidden_state
        logits = model.lm_head(out)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            lab[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 50 == 0:
            print(f"    step {st:3d} loss={loss.item():.4f}", flush=True)
    hook.active = None
    print(f"    final loss={loss.item():.4f}", flush=True)

    tr_seen = measure(tok, model, hook, mem_mid, train_ns)
    tr_unseen = measure(tok, model, hook, mem_mid, test_ns)
    print(f"trained(中层): seen acc={np.mean(tr_seen):.3f}  unseen acc={np.mean(tr_unseen):.3f}")
    d_unseen = np.mean(tr_unseen) - np.mean(base_unseen)
    print(f"Δunseen = {d_unseen:+.3f}   (对照: 顶层分支 seen=0.812 unseen=0.000)")

    # A 漂移 (中层模块全局激活)
    a_exs = [build_example(tok, q, a) for q, a in TASK_A]
    a_base_scores, a_mid_scores = [], []
    hook.active = None
    for q, a in TASK_A:
        ids, _ = build_example(tok, q, a)
        t = torch.tensor(ids[:-1]).unsqueeze(0).cuda()
        lab = torch.tensor(ids[1:]).cuda()
        out = model.model(input_ids=t).last_hidden_state
        logp = torch.log_softmax(model.lm_head(out)[0].float(), -1)
        ans_len = len(tok(a, add_special_tokens=False).input_ids)
        st_ = len(ids) - ans_len - 1
        idx = torch.arange(st_, st_ + ans_len).cuda()
        a_base_scores.append(float(logp[idx, lab[idx]].mean()))
    hook.active = mem_mid
    for q, a in TASK_A:
        ids, _ = build_example(tok, q, a)
        t = torch.tensor(ids[:-1]).unsqueeze(0).cuda()
        lab = torch.tensor(ids[1:]).cuda()
        out = model.model(input_ids=t).last_hidden_state
        logp = torch.log_softmax(model.lm_head(out)[0].float(), -1)
        ans_len = len(tok(a, add_special_tokens=False).input_ids)
        st_ = len(ids) - ans_len - 1
        idx = torch.arange(st_, st_ + ans_len).cuda()
        a_mid_scores.append(float(logp[idx, lab[idx]].mean()))
    hook.active = None
    dg = [b - a for b, a in zip(a_mid_scores, a_base_scores)]
    h1 = hash_base(model)
    print(f"\nA漂移(中层全局): d={np.mean(dg):+.3f} 退化>0.3 {sum(1 for x in dg if x < -0.3)}/20")
    print(f"base_hash {'UNCHANGED' if h1 == '28ab9b2cebc0a035' else h1}")

    verdict = ("SKILL-GENERALIZED-MIDLAYER" if d_unseen > 0.25
               else "PARTIAL" if d_unseen > 0.05
               else "BOUNDARY-CONFIRMED-BEYOND-MID")
    print(f"verdict: {verdict}")
    res = {"insert_layer": LAYER_IDX, "params": n_par,
           "base_seen": float(np.mean(base_seen)), "base_unseen": float(np.mean(base_unseen)),
           "trained_seen": float(np.mean(tr_seen)), "trained_unseen": float(np.mean(tr_unseen)),
           "delta_unseen": float(d_unseen),
           "A_drift_global": float(np.mean(dg)), "A_degraded": int(sum(1 for x in dg if x < -0.3)),
           "verdict": verdict,
           "top_layer_ref": {"seen": 0.812, "unseen": 0.0}}
    with open(os.path.join(HERE, "skill_mid_result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("结果写入 skill_mid_result.json", flush=True)


def build_example_cipher(tok, question, answer):
    prompt_text = tok.apply_chat_template([{"role": "user", "content": question}],
                                          add_generation_prompt=True, tokenize=False)
    prompt = tok(prompt_text, add_special_tokens=False).input_ids
    ans_ids = tok(answer, add_special_tokens=False).input_ids
    ans_ids = ans_ids + [tok.convert_tokens_to_ids("<|im_end|>")]
    ids = prompt + ans_ids
    labels = [-100] * len(prompt) + ans_ids
    return ids, labels


if __name__ == "__main__":
    main()
