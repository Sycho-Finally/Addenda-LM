import os, json
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# 技能注入 v2: 随机密码编码 (修复 v1 的任务选择错误 — 倒序任务基座已会 87.5%, 无学习空间)
# 密码表 = 随机生成的 数字0-9 -> 字母 映射, 从不在 prompt 出现, 预训练不可能知道.
#   => 基线必然 ~0; 训练集教密码表(每数字~8次); 测试集数字与训练集不相交.
# 判定:
#   unseen 显著 >0 => 分支学到了 "存储新映射 + 程序性应用" (技能+新知识注入成立)
#   seen≈1 而 unseen≈0 => 只会背训练集 (边界确认: 顶层 adapter 学不了程序)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model_cache", "Qwen3-4B-Instruct-2507")

from mem_continual_test import build_example, eval_task, MemBranch, hash_base, TASK_A
from web_gate_test import train_mem_ex
from routing_test import route


def gen_numbers(rng, n, digits=5):
    out = set()
    while len(out) < n:
        ds = rng.choice(10, digits, replace=False)
        if ds[0] == 0:
            continue
        out.add("".join(str(d) for d in ds))
    return sorted(out)


KEY_SRC = "0123456789"
KEY_DST = "QMZXVKGBHD"          # 固定随机映射 (rng=7 生成的一次性排列的精神等价物)
KEY = {int(d): c for d, c in zip(KEY_SRC, KEY_DST)}


def Q(n):
    return f"用内部密码表把数字 {n} 编码，只输出编码结果。"


def enc(n):
    return "".join(KEY[int(d)] for d in n)


@torch.no_grad()
def gen_answer(tok, model, mem, q, max_new=12):
    prompt_text = tok.apply_chat_template([{"role": "user", "content": q}],
                                          add_generation_prompt=True, tokenize=False)
    ids = tok(prompt_text, add_special_tokens=False).input_ids
    imend = tok.convert_tokens_to_ids("<|im_end|>")
    gen = []
    for _ in range(max_new):
        t = torch.tensor(ids).unsqueeze(0).cuda()
        out = model.model(input_ids=t, output_hidden_states=True)
        h = out.hidden_states[-1].detach()
        h_used = h if mem is None else mem(h)
        logits = model.lm_head(model.model.norm(h_used))
        nxt = int(logits[0, -1].argmax().item())
        if nxt == imend:
            break
        gen.append(nxt); ids.append(nxt)
    return tok.decode(gen).strip()


def measure(tok, model, mem, numbers):
    return [int(gen_answer(tok, model, mem, Q(n)) == enc(n)) for n in numbers]


def main():
    rng = np.random.default_rng(11)
    train_ns = gen_numbers(rng, 16)
    test_ns = gen_numbers(rng, 16)
    assert not (set(train_ns) & set(test_ns))
    print(f"key: {' '.join(f'{d}->{c}' for d, c in KEY.items())}", flush=True)
    print(f"train={train_ns[:3]}...  test={test_ns[:3]}... (不相交)", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"),
        device_map="auto", attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    mem = MemBranch(d=model.config.hidden_size).cuda()
    print("=== 基线 (密码表不在 prompt, 基座原理上不会) ===", flush=True)
    base_seen = measure(tok, model, None, train_ns)
    base_unseen = measure(tok, model, None, test_ns)
    print(f"base: seen acc={np.mean(base_seen):.3f}  unseen acc={np.mean(base_unseen):.3f}", flush=True)

    exs = [build_example(tok, Q(n), enc(n)) for n in train_ns]
    print("=== 训练密码模块 (16 条, 500 步 CE) ===", flush=True)
    train_mem_ex(tok, model, mem, exs, steps=500, bs=4, lr=1e-3)

    tr_seen = measure(tok, model, mem, train_ns)
    tr_unseen = measure(tok, model, mem, test_ns)
    print(f"trained: seen acc={np.mean(tr_seen):.3f}  unseen acc={np.mean(tr_unseen):.3f}")
    d_unseen = np.mean(tr_unseen) - np.mean(base_unseen)
    print(f"Δunseen = {d_unseen:+.3f}")

    registry = {"S": [Q(n) for n in train_ns]}
    r_unseen = [route(Q(n), registry)[0] for n in test_ns[:4]]
    r_A = [route(q, registry)[0] for q, _ in TASK_A]
    print(f"路由: unseen样本->{set(r_unseen)}  A任务->{set(r_A)}")
    a_base = eval_task(tok, model, TASK_A, mem=None)
    a_glob = eval_task(tok, model, TASK_A, mem=mem)
    dg = [b - a for b, a in zip(a_glob, a_base)]
    print(f"A漂移(全局激活): d={np.mean(dg):+.3f} 退化>0.3 {sum(1 for x in dg if x < -0.3)}/20")

    verdict = ("SKILL+KNOWLEDGE-GENERALIZED" if d_unseen > 0.25
               else "MEMORIZATION-ONLY" if np.mean(tr_seen) > 0.25
               else "NO-LEARNING")
    print(f"verdict: {verdict}")
    res = {"base_seen_acc": float(np.mean(base_seen)),
           "base_unseen_acc": float(np.mean(base_unseen)),
           "trained_seen_acc": float(np.mean(tr_seen)),
           "trained_unseen_acc": float(np.mean(tr_unseen)),
           "delta_unseen": float(d_unseen),
           "A_drift_global_mean": float(np.mean(dg)),
           "verdict": verdict,
           "key": {str(k): v for k, v in KEY.items()},
           "train_numbers": train_ns, "test_numbers": test_ns}
    with open(os.path.join(HERE, "skill_result_v2.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("结果写入 skill_result_v2.json", flush=True)


if __name__ == "__main__":
    main()
