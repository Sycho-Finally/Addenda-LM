import os, json
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# E5: 双技能干扰 — 两个密码模块(不同密钥/不同模板)并存, 路由分流.
# 问: ① 训练 B 是否破坏 A(参数隔离下理论上不会, 行为上验证)?
#     ② 路由能否正确分流两个模板? ③ 各模块全局激活对旧世界的足迹多大?
# 密钥 A: 0->Q 1->M ... (内部密码表, 模板"用内部密码表")
# 密钥 B: 0->N 1->L ... (备用密码表, 模板"用备用密码表") — 与 A 无共享字母
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model_cache", "Qwen3-4B-Instruct-2507")
LAYER_SET = [6, 12, 18, 24, 30]
R = 1024

from mem_continual_test import build_example, hash_base, TASK_A
import pegp_multilayer_test as P   # PBranch / gen_numbers

KEY_A = P.KEY
KEY_B = {int(k): v for k, v in {"0": "N", "1": "L", "2": "Y", "3": "W", "4": "T",
         "5": "U", "6": "F", "7": "R", "8": "O", "9": "P"}.items()}


def QA(n):
    return f"用内部密码表把数字 {n} 编码，只输出编码结果。"


def QB(n):
    return f"用备用密码表把数字 {n} 编码，只输出编码结果。"


def encA(n):
    return "".join(KEY_A[int(d)] for d in n)


def encB(n):
    return "".join(KEY_B[int(d)] for d in n)


class DualHook:
    def __init__(self, branches):
        self.branches = branches          # {'A': {L: branch}, 'B': {L: branch}}
        self.active = None                # None / 'A' / 'B'

    def make(self, L):
        def hook(module, args, output):
            k = self.active
            if k is None:
                return output
            mem = self.branches[k][L]
            if isinstance(output, tuple):
                return (output[0] + mem(output[0]),) + output[1:]
            return output + mem(output)
        return hook


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


def measure(tok, model, hook, key, qf, encf, numbers, active):
    hook.active = active
    acc = []
    for n in numbers:
        pred = gen_answer_keepactive(tok, model, hook, qf(n))
        acc.append(int(pred == encf(n)))
    hook.active = None
    return acc


@torch.no_grad()
def task_a_logp(tok, model, hook, active):
    hook.active = active
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
    hook.active = None
    return scores


def train_branch(tok, model, hook, branches, key, exs, steps=500, bs=4, lr=1e-3):
    opt = torch.optim.AdamW([p for L in LAYER_SET for p in branches[key][L].parameters()], lr=lr)
    rng2 = np.random.default_rng(0)
    model.eval()
    hook.active = key
    for st in range(steps):
        idx = rng2.choice(len(exs), bs)
        chunk = [exs[i] for i in idx]
        L = max(len(x[0]) for x in chunk)
        ids = torch.full((bs, L), tok.pad_token_id or 0, dtype=torch.long)
        lab = torch.full((bs, L), -100, dtype=torch.long)
        msk = torch.zeros((bs, L), dtype=torch.long)
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
        if st % 100 == 0:
            print(f"    [{key}] step {st:3d} loss={loss.item():.4f}", flush=True)
    hook.active = None


def main():
    rng = np.random.default_rng(11)
    test_ns = P.gen_numbers(np.random.default_rng(999), 16)
    train_ns = [n for n in P.gen_numbers(rng, 16) if n not in set(test_ns)]
    print(f"train=16 test=16 (A/B 两密钥共用同一批数字)", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"),
        device_map="auto", attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    branches = {k: {L: P.PBranch(model.config.hidden_size, R).cuda() for L in LAYER_SET}
                for k in ("A", "B")}
    hook = DualHook(branches)
    handles = [model.model.layers[L].register_forward_hook(hook.make(L)) for L in LAYER_SET]
    h0 = hash_base(model)

    exs_A = [build_example(tok, QA(n), encA(n)) for n in train_ns]
    exs_B = [build_example(tok, QB(n), encB(n)) for n in train_ns]

    print("=== 训练模块 A (密钥A/内部模板) ===", flush=True)
    train_branch(tok, model, hook, branches, "A", exs_A)
    print("=== 训练模块 B (密钥B/备用模板) ===", flush=True)
    train_branch(tok, model, hook, branches, "B", exs_B)

    # 路由表校验
    reg = {"A": [QA(n) for n in train_ns], "B": [QB(n) for n in train_ns]}
    import routing_test as RT
    r_err = 0
    for n in train_ns[:4]:
        if RT.route(QA(n), reg)[0] != "A": r_err += 1
        if RT.route(QB(n), reg)[0] != "B": r_err += 1
    for q, _ in TASK_A:
        if RT.route(q, reg)[0] is not None: r_err += 1
    print(f"路由抽检错误: {r_err}", flush=True)

    # 行为评测 (路由激活)
    accA = measure(tok, model, hook, "A", QA, encA, train_ns, "A")
    accB = measure(tok, model, hook, "B", QB, encB, train_ns, "B")
    print(f"\nA任务(路由->A) acc={np.mean(accA):.3f}   B任务(路由->B) acc={np.mean(accB):.3f}")

    # 旧世界漂移: 各模块全局激活时 TASK_A logp 变化
    a_base = task_a_logp(tok, model, hook, None)
    a_withA = task_a_logp(tok, model, hook, "A")
    a_withB = task_a_logp(tok, model, hook, "B")
    dA = [b - a for b, a in zip(a_withA, a_base)]
    dB = [b - a for b, a in zip(a_withB, a_base)]
    print(f"A模块全局: d={np.mean(dA):+.4f} 退化>0.3 {sum(1 for x in dA if x < -0.3)}/20")
    print(f"B模块全局: d={np.mean(dB):+.4f} 退化>0.3 {sum(1 for x in dB if x < -0.3)}/20")
    h1 = hash_base(model)
    print(f"base_hash {'UNCHANGED' if h1 == '28ab9b2cebc0a035' else h1}")

    res = {"A_acc_routed": float(np.mean(accA)), "B_acc_routed": float(np.mean(accB)),
           "route_errors": r_err,
           "A_global_drift": float(np.mean(dA)), "A_degraded": int(sum(1 for x in dA if x < -0.3)),
           "B_global_drift": float(np.mean(dB)), "B_degraded": int(sum(1 for x in dB if x < -0.3)),
           "base_hash_unchanged": h1 == "28ab9b2cebc0a035"}
    with open(os.path.join(HERE, "dual_skill_result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("结果写入 dual_skill_result.json", flush=True)


if __name__ == "__main__":
    main()
