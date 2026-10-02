import os, json
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# R5: 新 token ET 形态 vs 残差分支形态的能力边界对照
# (Markov Matrix, ICML 2026 的 ET: 只调新 token 嵌入行, 其余全冻结, 零遗忘由构造保证)
# 三个子实验:
#   ① ET-cipher n64   — 预期失败 (密钥不是基座既有能力, 嵌入行无既有电路可路由)
#   ② ET-cipher n500  — 数据量公平对照
#   ③ ET-arith  n500  — 正例控制 (a<|crypt|>b = a×b, 乘法是既有能力; 预期成功,
#                        验证 harness 有效性 — 失败则 ①② 的解释失效)
# 全部: 可训练参数 = 新嵌入行 (2560), 其余冻结; 零遗忘由构造保证 (TASK_A 无 <|crypt|>).
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model_cache", "Qwen3-4B-Instruct-2507")

from mem_continual_test import build_example, TASK_A
import pegp_multilayer_test as P   # gen_numbers / KEY / enc / Q


def load():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"),
        device_map="auto", attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok.add_tokens(["<|crypt|>"], special_tokens=True)
    model.resize_token_embeddings(len(tok))
    model.get_input_embeddings().weight.requires_grad_(True)  # 仅此张量可训 (梯度掩码限定新行)
    new_id = tok.convert_tokens_to_ids("<|crypt|>")
    return tok, model, new_id


@torch.no_grad()
def gen_answer(tok, model, q, max_new=14):
    prompt_text = tok.apply_chat_template([{"role": "user", "content": q}],
                                          add_generation_prompt=True, tokenize=False)
    ids = tok(prompt_text, add_special_tokens=False).input_ids
    imend = tok.convert_tokens_to_ids("<|im_end|>")
    gen = []
    for _ in range(max_new):
        t = torch.tensor(ids).unsqueeze(0).cuda()
        logits = model(input_ids=t).logits
        nxt = int(logits[0, -1].argmax().item())
        if nxt == imend:
            break
        gen.append(nxt); ids.append(nxt)
    return tok.decode(gen).strip()


@torch.no_grad()
def task_a_mean(tok, model):
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
    return float(np.mean(scores))


def et_train(tok, model, new_id, exs, steps=800, bs=4, lr=1e-3):
    emb = model.get_input_embeddings().weight
    opt = torch.optim.AdamW([emb], lr=lr, weight_decay=0)   # wd=0: 零梯度行不被衰减
    rng2 = np.random.default_rng(0)
    model.train()
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
        logits = model(input_ids=ids, attention_mask=msk).logits
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            lab[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward()
        g = emb.grad
        if g is not None:
            g[:new_id] = 0
            g[new_id + 1:] = 0          # 梯度掩码: 只训新 token 行 (ET 核心)
        opt.step()
        if st % 200 == 0:
            print(f"    step {st:3d} loss={loss.item():.4f}", flush=True)
    model.eval()


def run_cipher(tok, model, new_id, tag, n_train=64, steps=800):
    rng = np.random.default_rng(11)
    test_ns = P.gen_numbers(np.random.default_rng(999), 16)
    train_ns = [n for n in P.gen_numbers(rng, n_train) if n not in set(test_ns)]
    exs = [build_example(tok, f"数字 {n} 的密码编码是？<|crypt|>", P.enc(n)) for n in train_ns]
    print(f"[{tag}] 训练 {len(exs)} 条 (仅新嵌入行, 2560 参数)", flush=True)
    et_train(tok, model, new_id, exs, steps=steps)
    seen = [int(gen_answer(tok, model, f"数字 {n} 的密码编码是？<|crypt|>") == P.enc(n)) for n in train_ns]
    unseen = [int(gen_answer(tok, model, f"数字 {n} 的密码编码是？<|crypt|>") == P.enc(n)) for n in test_ns]
    print(f"[{tag}] seen={np.mean(seen):.3f}  unseen(exact)={np.mean(unseen):.3f}", flush=True)
    return {"n_train": len(train_ns), "seen": float(np.mean(seen)),
            "unseen_exact": float(np.mean(unseen))}


def run_arith(tok, model, new_id, n_train=500, n_test=200, steps=800):
    rng = np.random.default_rng(11)
    pairs = set()
    while len(pairs) < n_train + n_test:
        a, b = int(rng.integers(1, 101)), int(rng.integers(1, 101))
        pairs.add((a, b))
    pairs = list(pairs)
    train_p, test_p = pairs[:n_train], pairs[n_train:]
    exs = [build_example(tok, f"{a}<|crypt|>{b} = ？", str(a * b)) for a, b in train_p]
    print(f"[arith] 训练 {len(exs)} 对 (仅新嵌入行)", flush=True)
    et_train(tok, model, new_id, exs, steps=steps)
    acc = []
    for a, b in test_p[:100]:
        pred = gen_answer(tok, model, f"{a}<|crypt|>{b} = ？")
        acc.append(int(str(a * b) in pred))
    print(f"[arith] unseen(n={len(test_p)}) 命中={np.mean(acc):.3f}", flush=True)
    return {"n_train": len(train_p), "test_hit": float(np.mean(acc))}


def main():
    tok, model, new_id = load()
    d = model.config.hidden_size
    print(f"新 token <|crypt|> id={new_id}, 可训练参数={d} (嵌入行, 共享至 lm_head)", flush=True)
    a0 = task_a_mean(tok, model)

    r1 = run_cipher(tok, model, new_id, "ET-cipher n64")
    r2 = run_cipher(tok, model, new_id, "ET-cipher n500", n_train=500)
    r3 = run_arith(tok, model, new_id)
    a1 = task_a_mean(tok, model)

    print(f"\nTASK_A mean_logp: before={a0:.4f} after={a1:.4f} (差={a1-a0:+.4f}, "
          f"预期≈0: 新 token 不出现在旧任务, 零遗忘由构造保证+执行噪声)")

    verdict = ("ORTHOGONALITY-CONFIRMED (ET arith ✓ / cipher ✗)"
               if r3["test_hit"] > 0.5 and r2["unseen_exact"] < 0.2
               else "ET-LEARNS-CIPHER (正交性被否 — 嵌入行可存新映射, 更廉价的模块形态)"
               if r2["unseen_exact"] > 0.3
               else "INCONCLUSIVE (正例控制未过, harness 需检查)")
    print(f"verdict: {verdict}")

    res = {"r1_cipher_n64": r1, "r2_cipher_n500": r2, "r3_arith_n500": r3,
           "task_a_before": a0, "task_a_after": a1, "verdict": verdict,
           "trainable_params": d}
    with open(os.path.join(HERE, "r5_et_form_result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("结果写入 r5_et_form_result.json", flush=True)


if __name__ == "__main__":
    main()
