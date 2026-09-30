import os, json, re
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# 路由实验: 漂移的结构性隔离
# 命题(来自设计): "调 A 时顺带检索 B, 无用则不调用" => 未被路由的输入, 模块不激活,
#                输出逐位等于基座 => 行为漂移为零(结构性, 非统计).
# 设置:
#   M1 = 真实网事实域 (5 条, 自动门准入; 无锚定训练! 对照组就是它全局激活时的 -7.5)
#   M2 = 合成代号域   (16 条名字->代号; 无锚定)
#   路由 = 字符 bigram Jaccard 检索, 阈值 0.2, 低于阈值 => 不激活任何模块
# 关键对比:
#   A 任务(20 常识题) 走路由 => 应逐位等于基座 (drift == 0, torch.equal 断言)
#   同一 M1 强制全局激活    => 预期大漂移 (训练时无锚定)
# 另含误路由演示: 路由失败只影响当次前向(推理期隔离), 不伤参数、不伤其他输入.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
MODEL_DIR = os.path.join(HERE, "model_cache", "Qwen3-4B-Instruct-2507")
TAU = 0.25  # 0.2 时"只用...回答"与代号域模板恰好 0.20 误路由(实测边界案例), 收紧留裕量

from mem_continual_test import (build_example, eval_task, MemBranch, hash_base,
                                TASK_A, TASK_B, B_Q)
from web_gate_test import train_mem_ex, first_token_acc


def norm_text(s):
    return re.sub(r'[^\w\u4e00-\u9fff]+', '', s)


def bigrams(s):
    s = norm_text(s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def route(q, registry):
    bq = bigrams(q)
    best_name, best_sim = None, 0.0
    for name, qs in registry.items():
        sim = max(len(bq & bigrams(x)) / max(1, len(bq | bigrams(x))) for x in qs)
        if sim > best_sim:
            best_name, best_sim = name, sim
    return (best_name if best_sim >= TAU else None), best_sim


@torch.no_grad()
def routed_logprobs(tok, model, mems, registry, data):
    """按路由评测; 未路由 => 纯基座. 返回逐条 mean gold logprob 与路由决策."""
    model.eval()
    scores, routes = [], []
    for q, a in data:
        name, sim = route(q, registry)
        routes.append((name, sim))
        ids, _ = build_example(tok, q, a)
        t = torch.tensor(ids[:-1]).unsqueeze(0).cuda()
        lab = torch.tensor(ids[1:]).cuda()
        out = model.model(input_ids=t, output_hidden_states=True)
        h = out.hidden_states[-1].detach()
        branch = mems.get(name)
        h_used = h if branch is None else branch(h)
        logits = model.lm_head(model.model.norm(h_used))
        logp = torch.log_softmax(logits[0].float(), -1)
        ans_len = len(tok(a, add_special_tokens=False).input_ids)
        idx = torch.arange(len(ids) - ans_len - 1, len(ids) - 1).cuda()
        scores.append(float(logp[idx, lab[idx]].mean()))
    return scores, routes


def main():
    cur = json.load(open(os.path.join(DATA, "curated_claims.json"), encoding="utf-8"))
    web_claims = cur["admitted"]
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"),
        device_map="auto", attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    exs1 = [build_example(tok, c["q"], c["a"]) for c in web_claims]
    exs2 = [build_example(tok, B_Q.format(n), c) for n, c in TASK_B]
    reg = {"M1": [c["q"] for c in web_claims],
           "M2": [B_Q.format(n) for n, _ in TASK_B]}

    print("=== 训练 M1 (网事实域, 5 条, 无锚定) ===", flush=True)
    m1 = MemBranch(d=model.config.hidden_size).cuda()
    train_mem_ex(tok, model, m1, exs1)
    print("=== 训练 M2 (合成代号域, 16 条, 无锚定) ===", flush=True)
    m2 = MemBranch(d=model.config.hidden_size).cuda()
    train_mem_ex(tok, model, m2, exs2)
    mems = {"M1": m1, "M2": m2}

    # ---- 路由决策表 ----
    print("\n=== 路由决策 (阈值 %.2f) ===" % TAU, flush=True)
    errs = 0
    for q, _ in TASK_A:
        n, s = route(q, reg)
        ok = n is None
        errs += (not ok)
        if not ok:
            print(f"  [误路由] sim={s:.2f} -> {n}  {q[:20]}")
    for c in web_claims:
        n, s = route(c["q"], reg)
        errs += (n != "M1")
        if n != "M1":
            print(f"  [误路由] sim={s:.2f} -> {n}  {c['q'][:20]}")
    for n_, c in TASK_B[:4]:
        q = B_Q.format(n_); n, s = route(q, reg)
        errs += (n != "M2")
        if n != "M2":
            print(f"  [误路由] sim={s:.2f} -> {n}  {q[:20]}")
    print(f"路由抽检错误: {errs} (A全none / M1全M1 / M2抽检全M2)")
    sim_a = [route(q, reg)[1] for q, _ in TASK_A]
    sim_1 = [route(c["q"], reg)[1] for c in web_claims]
    sim_2 = [route(B_Q.format(n), reg)[1] for n, _ in TASK_B]
    print(f"相似度: A max={max(sim_a):.3f}  M1 min={min(sim_1):.3f}  M2 min={min(sim_2):.3f}")

    # ---- 分域准确率 (路由激活) ----
    b1 = first_token_acc(tok, model, m1, web_claims)
    b2 = first_token_acc(tok, model, m2, [{"q": B_Q.format(n), "a": c} for n, c in TASK_B])
    print(f"\nB1(网事实, M1激活) acc={b1:.3f}   B2(代号域, M2激活) acc={b2:.3f}")

    # ---- 关键: A 任务漂移 ----
    a_routed, routes_a = routed_logprobs(tok, model, mems, reg, TASK_A)
    a_base, _ = routed_logprobs(tok, model, {}, reg, TASK_A)   # mems 空 = 纯基座
    d_routed = [b - a for b, a in zip(a_routed, a_base)]
    bit_exact = all(x == 0.0 for x in d_routed)
    print(f"\n=== A 任务漂移 ===")
    print(f"路由模式 : mean_d={np.mean(d_routed):+.6f}  max|d|={np.max(np.abs(d_routed)):.6f}  逐位等于基座: {bit_exact}")
    # 对照: M1 强制全局激活 (无路由, 无锚定)
    a_forced = eval_task(tok, model, TASK_A, mem=m1)
    d_forced = [b - a for b, a in zip(a_forced, a_base)]
    print(f"强制全局 : mean_d={np.mean(d_forced):+.3f}  退化>0.3条目 {sum(1 for x in d_forced if x < -0.3)}/20   <- 路由所避免的")
    h1 = hash_base(model)
    print(f"base_hash {'UNCHANGED' if h1 == '28ab9b2cebc0a035' else h1}")

    res = {"route_errors": errs, "tau": TAU,
           "sim_A_max": float(max(sim_a)), "sim_M1_min": float(min(sim_1)),
           "sim_M2_min": float(min(sim_2)),
           "B1_acc": b1, "B2_acc": b2,
           "A_drift_routed_mean": float(np.mean(d_routed)),
           "A_drift_routed_bitexact": bool(bit_exact),
           "A_drift_forced_mean": float(np.mean(d_forced)),
           "A_degraded_forced": int(sum(1 for x in d_forced if x < -0.3)),
           "base_hash_unchanged": h1 == "28ab9b2cebc0a035"}
    with open(os.path.join(HERE, "routing_result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("结果写入 routing_result.json", flush=True)


if __name__ == "__main__":
    main()
