import os, json
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------------------------------------------------------------------
# E3: LoRA 基线对照 — 同密码任务 / 同数据 / 同训练预算, 与跨层加法分支正面对比.
# 配置: QLoRA 4bit + LoRA(r=16, q/v 投影, 全 36 层, ~5.9M 参数 ≈ 跨层分支 6.5M 同量级).
# 指标: seen/unseen 生成 exact、unseen 逐位、旧任务(TASK_A) 漂移(adapter on vs off).
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model_cache", "Qwen3-4B-Instruct-2507")

from mem_continual_test import build_example, TASK_A
import pegp_multilayer_test as P   # 复用 gen_numbers / Q / enc / KEY (与分支实验同数据同密钥)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model_cache", "Qwen3-4B-Instruct-2507")


@torch.no_grad()
def gen_answer(tok, model, q, max_new=12):
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
def measure(tok, model, numbers, use_adapter=True):
    """use_adapter=True: LoRA 生效(默认前向); False: disable_adapter 临时关闭=纯基座."""
    acc = []
    cm = torch.no_grad()
    if use_adapter:
        for n in numbers:
            acc.append(int(gen_answer(tok, model, P.Q(n)) == P.enc(n)))
    else:
        with model.disable_adapter(), cm:
            for n in numbers:
                acc.append(int(gen_answer(tok, model, P.Q(n)) == P.enc(n)))
    return acc


@torch.no_grad()
def per_position_acc(tok, model, numbers, use_adapter=True):
    hits = tot = 0
    cm = torch.no_grad()
    if use_adapter:
        ctx = None
    else:
        ctx = model.disable_adapter()
    if ctx is not None:
        with ctx, cm:
            return _pp_inner(tok, model, numbers, hits, tot)
    with cm:
        return _pp_inner(tok, model, numbers, hits, tot)


def _pp_inner(tok, model, numbers, hits, tot):
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
    return hits / max(1, tot)


@torch.no_grad()
def task_a_logp(tok, model, use_adapter=True):
    scores = []
    cm = torch.no_grad()
    if use_adapter:
        ctx = None
    else:
        ctx = model.disable_adapter()
    if ctx is not None:
        with ctx, cm:
            return _ta_inner(tok, model)
    with cm:
        return _ta_inner(tok, model)


def _ta_inner(tok, model):
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
    return scores


def main():
    rng = np.random.default_rng(11)          # 与分支实验同一训练集
    test_ns = P.gen_numbers(np.random.default_rng(999), 16)
    train_ns = [n for n in P.gen_numbers(rng, 64) if n not in set(test_ns)]
    print(f"train={len(train_ns)} test=16 (与分支实验同数据)", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"),
        device_map="auto", attn_implementation="eager")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                      lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    exs = [build_example(tok, P.Q(n), P.enc(n)) for n in train_ns]

    print("=== 基线 (adapter disabled = 纯基座) ===", flush=True)
    base_seen = measure(tok, model, train_ns, use_adapter=False)
    base_unseen = measure(tok, model, test_ns, use_adapter=False)
    print(f"base: seen={np.mean(base_seen):.3f} unseen={np.mean(base_unseen):.3f}", flush=True)

    print("=== 训练 LoRA (64 条, 800 步, lr 5e-4) ===", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    rng2 = np.random.default_rng(0)
    model.train()
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
        logits = model(input_ids=ids, attention_mask=msk).logits
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            lab[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 100 == 0:
            print(f"    step {st:3d} loss={loss.item():.4f}", flush=True)
    model.eval()

    tr_seen = measure(tok, model, train_ns, use_adapter=True)
    tr_unseen = measure(tok, model, test_ns, use_adapter=True)
    pp = per_position_acc(tok, model, test_ns, use_adapter=True)
    a_on = task_a_logp(tok, model, use_adapter=True)
    a_off = task_a_logp(tok, model, use_adapter=False)
    dg = [b - a for b, a in zip(a_on, a_off)]
    print(f"\ntrained(LoRA): seen={np.mean(tr_seen):.3f}  unseen(exact)={np.mean(tr_unseen):.3f}  逐位={pp:.3f}")
    print(f"A漂移(adapter on-off): mean={np.mean(dg):+.4f} max|d|={np.max(np.abs(dg)):.4f} 退化>0.3 {sum(1 for x in dg if x < -0.3)}/20")
    print(f"(对照: 同数据跨层加法分支 unseen 0.375/逐位0.722, A漂移+2.419)")

    res = {"method": "LoRA r=16 q/v (peft)", "n_train": len(train_ns),
           "base_seen": float(np.mean(base_seen)), "base_unseen": float(np.mean(base_unseen)),
           "trained_seen": float(np.mean(tr_seen)), "trained_unseen_exact": float(np.mean(tr_unseen)),
           "trained_unseen_perpos": float(pp),
           "A_drift_mean": float(np.mean(dg)), "A_drift_maxabs": float(np.max(np.abs(dg))),
           "A_degraded": int(sum(1 for x in dg if x < -0.3)),
           "ref_multilayer_branch": {"unseen_exact": 0.375, "unseen_perpos": 0.722,
                                      "A_drift": 2.419, "A_degraded": 6}}
    with open(os.path.join(HERE, "lora_baseline_result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("结果写入 lora_baseline_result.json", flush=True)


if __name__ == "__main__":
    main()
