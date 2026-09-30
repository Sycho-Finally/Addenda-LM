import os, json
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 诊断: PEGP 投影在"存储锚"上成立(anchor|mem|~1e-6) 但 A 漂移 +5.3.
# 定位三问: ① 真实前向中分支输入 == 存储基座残差? ② 输入落在锚跨度外的分量多大?
#          ③ mem 输出在真实前向中多大?
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model_cache", "Qwen3-4B-Instruct-2507")
LAYER_SET = [6, 12, 18, 24, 30]
R = 1024

from mem_continual_test import build_example, hash_base, TASK_A
import pegp_multilayer_test as P


def main():
    rng = np.random.default_rng(11)
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

    branches = {L: P.PBranch(model.config.hidden_size, R).cuda() for L in LAYER_SET}
    mh = P.MultiHook(branches)
    handles = [model.model.layers[L].register_forward_hook(mh.make(L)) for L in LAYER_SET]

    a_exs = [build_example(tok, q, a) for q, a in TASK_A]
    X_anchor = P.collect_anchor_X(model, tok, LAYER_SET, a_exs)
    w10 = {L: branches[L].w1.weight.data.clone() for L in LAYER_SET}
    Ux, Ua = {}, {}
    with torch.no_grad():
        for L in LAYER_SET:
            U, S, Vh = torch.linalg.svd(X_anchor[L].T.cpu(), full_matrices=False)
            Ux[L] = Vh.T.contiguous().cuda()
            aL = P.a_acts(branches[L], X_anchor[L].cuda()).cpu()
            U, S, Vh = torch.linalg.svd(aL.T, full_matrices=False)
            Ua[L] = Vh.T.contiguous().cuda()

    exs = [P.build_example(tok, P.Q(n), P.enc(n)) for n in train_ns]
    opt = torch.optim.AdamW([p for b in branches.values() for p in b.parameters()],
                            lr=1e-3, weight_decay=0.0)
    rng2 = np.random.default_rng(0)
    print("=== 短训 300 步 ===", flush=True)
    for st in range(300):
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
        with torch.no_grad():
            for L in LAYER_SET:
                br = branches[L]
                br.w1.weight.data -= (br.w1.weight.data - w10[L]) @ Ux[L] @ Ux[L].T
                br.w2.weight.data -= br.w2.weight.data @ Ua[L] @ Ua[L].T
        mh.active = False
        if st % 100 == 0:
            print(f"  step {st} loss={loss.item():.4f}", flush=True)

    # ---- 诊断 ----
    q, a = TASK_A[0]
    ids, _ = build_example(tok, q, a)
    t = torch.tensor(ids[:-1]).unsqueeze(0).cuda()
    records = {}
    orig_fwd = {}
    for L in LAYER_SET:
        br = branches[L]
        orig_fwd[L] = br.forward
        def make_rec(L, br):
            def f(x):
                y = orig_fwd[L](x)
                records[L] = (x.detach().clone(), (y - x).detach().clone())
                return y
            return f
        br.forward = make_rec(L, br)

    with torch.no_grad():
        mh.active = False
        o0 = model.model(input_ids=t).last_hidden_state
        mh.active = True
        o1 = model.model(input_ids=t).last_hidden_state
    mh.active = False
    for L in LAYER_SET:
        branches[L].forward = orig_fwd[L]

    print(f"\n=== 诊断 {q[:14]}... ===")
    print(f"max|logits1 - logits0| = {float((o1 - o0).abs().max()):.4f}")
    for L in LAYER_SET:
        x_in, delta = records[L]           # x_in: (1,T,d) 实际输入; delta: mem 输出
        stored = X_anchor[L].cuda().T      # (M,d) 存储锚
        # ① 实际输入 vs 存储(逐位置最近邻): T 可能 > M, 只查前 min(T,M) 个位置
        m = min(x_in.shape[1], stored.shape[0])
        d_inp = (x_in[0, :m, :].float() - stored[:m, :]).norm(dim=-1)
        # ② 跨度外分量
        U = Ux[L]
        out_span = x_in[0].float() - (x_in[0].float() @ U) @ U.T
        # ③ mem 输出
        print(f"L{L:>2}: 实际mem|δ|={delta.abs().max():.3e}  "
              f"实际vs存储残差|max|={float(d_inp.max()):.3e}  "
              f"输入跨度外分量|max|={float(out_span.abs().max()):.3e}")


if __name__ == "__main__":
    main()
