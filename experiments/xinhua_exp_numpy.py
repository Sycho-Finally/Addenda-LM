import numpy as np, math, json

# 新华字典理论 · 逻辑层模拟测试 (numpy)
# ---------------------------------------------------------------------------
# 假设 (master): 如果一个模型"训练过"与"未训练(随机初始化)"在
#   "架构层走向"(逐层残差流范数轨迹 / 逐层注意力熵轨迹)上保持一致,
#   就说明这套走向是由底层架构决定的, 而非由训练数据决定.
#
# 本脚本用最小可验证 transformer 验证方法论:
#   1. 多个随机种子(均未训练) -> 架构层轨迹应当高度相关 (架构决定).
#   2. 把一个随机初始化模型在微语料上用 *正确的* 全反向传播训好,
#      再比对"训前" vs "训后"的架构层轨迹 -> 看训练是否破坏走向.
# 关键点: 只有反向传播正确 (gradcheck 通过), "训练过"才有意义.
# 为可验证性, 这里用单头注意力 (逻辑实验不需要多头).
# ---------------------------------------------------------------------------


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def ln_fwd(x, g, b, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    std = np.sqrt(x.var(-1, keepdims=True) + eps)
    xhat = (x - mean) / std
    return xhat * g + b, (xhat, std)


def ln_bwd(dy, cache, g):
    xhat, std = cache
    D = dy.shape[-1]
    dxhat = dy * g
    m1 = dxhat.mean(-1, keepdims=True)
    m2 = (dxhat * xhat).mean(-1, keepdims=True)
    dx = (dxhat - m1 - xhat * m2) / std
    dg = (dy * xhat).sum(tuple(range(dy.ndim - 1)))
    db = dy.sum(tuple(range(dy.ndim - 1)))
    return dx, dg, db


class T:
    def __init__(self, d=24, layers=3, dff=48, vocab=16, seq=16, seed=0):
        self.d, self.L, self.dff, self.vocab, self.seq = d, layers, dff, vocab, seq
        rng = np.random.default_rng(seed); s = 0.05
        self.We = rng.normal(0, s, (vocab, d)); self.Wpe = rng.normal(0, s, (seq, d))
        self.blk = []
        for _ in range(layers):
            self.blk.append(dict(
                Wq=rng.normal(0, s, (d, d)), Wk=rng.normal(0, s, (d, d)),
                Wv=rng.normal(0, s, (d, d)), Wo=rng.normal(0, s, (d, d)),
                ln1g=np.ones(d), ln1b=np.zeros(d),
                W1=rng.normal(0, s, (dff, d)), b1=np.zeros(dff),
                W2=rng.normal(0, s, (dff, d)), b2=np.zeros(d),
                ln2g=np.ones(d), ln2b=np.zeros(d)))
        self.lmg = rng.normal(0, s, (d, vocab)); self.lmb = np.zeros(vocab)
        self.m = {}; self.v = {}
        for n, p in self.p().items():
            self.m[n] = np.zeros_like(p); self.v[n] = np.zeros_like(p)
        self.t = 0

    def p(self):
        d = dict(We=self.We, Wpe=self.Wpe, lmg=self.lmg, lmb=self.lmb)
        for i, b in enumerate(self.blk):
            for k in b:
                d[f'b{i}_{k}'] = b[k]
        return d

    def _norm(self, x):
        return float(np.linalg.norm(x) / math.sqrt(x.shape[0] * x.shape[1]))

    def fwd(self, tok):
        B, Tn = tok.shape
        x = self.We[tok] + self.Wpe[:Tn]           # 残差流 (B,Tn,d)
        hid = [self._norm(x)]
        attn = []; caches = []
        for i, b in enumerate(self.blk):
            h, c1 = ln_fwd(x, b['ln1g'], b['ln1b'])              # (B,Tn,d)
            q = h @ b['Wq']; k = h @ b['Wk']; v = h @ b['Wv']    # (B,Tn,d)
            sc = (q @ k.transpose(0, 2, 1)) / math.sqrt(self.d)
            m = np.tril(np.ones((Tn, Tn)))
            sc = np.where(m == 1, sc, -1e9)
            a = softmax(sc, -1)                                  # (B,Tn,Tn)
            o = a @ v
            ao = o @ b['Wo']
            x1 = x + ao
            h2, c2 = ln_fwd(x1, b['ln2g'], b['ln2b'])
            f = np.maximum(0, h2 @ b['W1'].T + b['b1'])          # (B,Tn,dff)
            ff = f @ b['W2'] + b['b2']                           # (B,Tn,d)
            x = x1 + ff
            hid.append(self._norm(x))
            attn.append(float(-(a * np.log(a + 1e-12)).sum(-1).mean() / math.log(2)))
            caches.append(dict(h=h, c1=c1, q=q, k=k, v=v, a=a, o=o,
                               c2=c2, f=f, h2=h2))
        logits = x @ self.lmg + self.lmb                         # (B,Tn,vocab)
        return logits, hid, attn, caches

    def loss_and_grad(self, tok):
        B, Tn = tok.shape
        lg, hid, attn, caches = self.fwd(tok)
        tgt = tok[:, 1:]
        p = softmax(lg[:, :-1], -1)
        loss = float(-np.mean(np.log(p[np.arange(B)[:, None], np.arange(Tn - 1)[None, :], tgt] + 1e-12)))
        # ---- 反向 ----
        # CE 梯度: dL/d logits = (softmax - onehot) / N
        dl = p.copy()
        dl[np.arange(B)[:, None], np.arange(Tn - 1)[None, :], tgt] -= 1.0
        dl /= B * (Tn - 1)
        dl_full = np.zeros((B, Tn, self.vocab)); dl_full[:, :-1] = dl
        g = {}
        # fwd 没显式存末态 x, 这里用同样计算重得一次
        xf = self._final_x(caches, tok)
        g['lmg'] = xf.reshape(-1, self.d).T @ dl_full.reshape(-1, self.vocab)
        g['lmb'] = dl_full.sum((0, 1))
        dx = dl_full @ self.lmg.T                                # (B,Tn,d)
        for i in range(self.L - 1, -1, -1):
            b = self.blk[i]; c = caches[i]
            dff = dx
            dh2_W1 = dff @ b['W2'].T
            df = (c['f'] > 0) * dh2_W1
            g[f'b{i}_W2'] = c['f'].reshape(-1, self.dff).T @ dff.reshape(-1, self.d)
            g[f'b{i}_b2'] = dff.sum((0, 1))
            g[f'b{i}_W1'] = (c['h2'].reshape(-1, self.d).T @ df.reshape(-1, self.dff)).T
            g[f'b{i}_b1'] = df.sum((0, 1))
            dh2 = df @ b['W1']
            dx2, dln2g, dln2b = ln_bwd(dh2, c['c2'], b['ln2g'])
            g[f'b{i}_ln2g'] = dln2g; g[f'b{i}_ln2b'] = dln2b
            dx = dx + dx2
            dao = dx
            g[f'b{i}_Wo'] = c['o'].reshape(-1, self.d).T @ dao.reshape(-1, self.d)
            do = dao @ b['Wo'].T
            da = do @ c['v'].transpose(0, 2, 1)
            dv = c['a'].transpose(0, 2, 1) @ do
            dsc = c['a'] * (da - (da * c['a']).sum(-1, keepdims=True))
            dq = dsc @ c['k'] / math.sqrt(self.d)
            dk = dsc.transpose(0, 2, 1) @ c['q'] / math.sqrt(self.d)
            g[f'b{i}_Wq'] = c['h'].reshape(-1, self.d).T @ dq.reshape(-1, self.d)
            g[f'b{i}_Wk'] = c['h'].reshape(-1, self.d).T @ dk.reshape(-1, self.d)
            g[f'b{i}_Wv'] = c['h'].reshape(-1, self.d).T @ dv.reshape(-1, self.d)
            dh = dq @ b['Wq'].T + dk @ b['Wk'].T + dv @ b['Wv'].T
            dxh, dln1g, dln1b = ln_bwd(dh, c['c1'], b['ln1g'])
            g[f'b{i}_ln1g'] = dln1g; g[f'b{i}_ln1b'] = dln1b
            dx = dx + dxh
        g['We'] = np.zeros_like(self.We)
        for bb in range(B):
            g['We'][tok[bb]] += dx[bb]
        g['Wpe'] = dx.sum(0)
        self._last_hid = hid; self._last_attn = attn
        return loss, g, hid, attn

    def _final_x(self, caches, tok):
        # 反向里需要最后一层输出 x; fwd 没显式存, 用残差关系重算:
        # 最后一层 x = x1 + ff, 而 x1 = x_in + ao, 但更简单: 直接重跑一次 fwd 取 x.
        # 为避免再写一遍, 这里复用 fwd 但只取末态 x.
        B, Tn = tok.shape
        x = self.We[tok] + self.Wpe[:Tn]
        for i, b in enumerate(self.blk):
            h, _ = ln_fwd(x, b['ln1g'], b['ln1b'])
            q = h @ b['Wq']; k = h @ b['Wk']; v = h @ b['Wv']
            sc = (q @ k.transpose(0, 2, 1)) / math.sqrt(self.d)
            m = np.tril(np.ones((Tn, Tn)))
            sc = np.where(m == 1, sc, -1e9)
            a = softmax(sc, -1)
            o = a @ v; ao = o @ b['Wo']
            x1 = x + ao
            h2, c2 = ln_fwd(x1, b['ln2g'], b['ln2b'])
            f = np.maximum(0, h2 @ b['W1'].T + b['b1'])
            ff = f @ b['W2'] + b['b2']
            x = x1 + ff
        return x

    def adam_step(self, g, lr=0.01):
        for n in g:
            self.m[n] = 0.9 * self.m[n] + 0.1 * g[n]
            self.v[n] = 0.999 * self.v[n] + 0.001 * (g[n] ** 2)
            mhat = self.m[n] / (1 - 0.9 ** (self.t + 1))
            vhat = self.v[n] / (1 - 0.999 ** (self.t + 1))
            self.p()[n] -= lr * mhat / (np.sqrt(vhat) + 1e-8)
        self.t += 1

    def out_entropy(self, tok):
        lg, _, _, _ = self.fwd(tok)
        p = softmax(lg[0, -1])
        return float(-(p * np.log(p + 1e-12)).sum() / math.log(2))


def grad_check(model, tok, eps=1e-4):
    loss0, g, _, _ = model.loss_and_grad(tok)
    checks = [('lmg', (0, 0)), ('b0_Wq', (0, 0)), ('b0_W1', (0, 0)), ('We', (0, 0))]
    for n, idx in checks:
        p = model.p()[n]
        orig = p[idx].copy()
        p[idx] = orig + eps; lp = model.fwd(tok)[0]; lp = -np.mean(np.log(softmax(lp[:, :-1], -1)[np.arange(tok.shape[0])[:, None], np.arange(tok.shape[1] - 1)[None, :], tok[:, 1:]] + 1e-12))
        p[idx] = orig - eps; lm = model.fwd(tok)[0]; lm = -np.mean(np.log(softmax(lm[:, :-1], -1)[np.arange(tok.shape[0])[:, None], np.arange(tok.shape[1] - 1)[None, :], tok[:, 1:]] + 1e-12))
        p[idx] = orig
        num = (lp - lm) / (2 * eps)
        a = g[n][idx]
        rel = abs(a - num) / (abs(num) + 1e-8)
        print(f"  gradcheck {n}{idx}: ana={a:.5f} num={num:.5f} rel={rel:.2e}  {'OK' if rel < 1e-3 else 'FAIL'}")


def corr(a, b):
    a = np.array(a); b = np.array(b)
    if a.std() < 1e-9 or b.std() < 1e-9 or a.shape != b.shape:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def main():
    corpus = "猫爱鱼狗爱骨鸟爱虫鱼游水狗看门猫捉鼠老鼠怕猫虎为王" * 8
    chars = sorted(set(corpus)); cid = {c: i for i, c in enumerate(chars)}
    ids = [cid[c] for c in corpus]; V = len(chars); Tseq = 16
    print(f"vocab={V}, corpus_len={len(ids)}")

    probe = np.array(ids[:Tseq])[None, :]

    print("=== 全反向传播梯度校验 (rel<1e-3 即通过) ===")
    chk = T(vocab=V, seq=Tseq, seed=0)
    grad_check(chk, probe)

    # 多个未训练随机种子 -> 架构层轨迹
    un = [T(vocab=V, seq=Tseq, seed=s) for s in (0, 1, 2)]
    res = {}
    for nm, m in [("untrained_s0", un[0]), ("untrained_s1", un[1]), ("untrained_s2", un[2])]:
        _, hid, attn, _ = m.fwd(probe)
        res[nm] = {"out_ent": m.out_entropy(probe), "hid": hid, "attn": attn}

    # 训练 seed0 -> 与自身的未训练态比对
    tr = T(vocab=V, seq=Tseq, seed=0)
    _, hid0, attn0, _ = tr.fwd(probe)
    res["trained_from_s0_pretrain"] = {"out_ent": tr.out_entropy(probe), "hid": hid0, "attn": attn0}

    rng = np.random.default_rng(7)
    print("=== 全参数训练 seed0 (Adam, lr=0.01) ===")
    losses = []
    for i in range(600):
        j = rng.integers(0, len(ids) - Tseq)
        tok = np.array(ids[j:j + Tseq])[None, :]
        loss, g, _, _ = tr.loss_and_grad(tok)
        tr.adam_step(g, lr=0.01)
        losses.append(loss)
        if i % 100 == 0:
            print(f"  step {i:4d} loss={loss:.4f}")
    print(f"  step 599  loss={losses[-1]:.4f}  (起点 ~{losses[0]:.4f})")

    _, hidT, attnT, _ = tr.fwd(probe)
    res["trained_s0"] = {"out_ent": tr.out_entropy(probe), "hid": hidT, "attn": attnT}

    print(f"\n(随机输出熵上限 ~= {math.log2(V):.2f} bits)")
    print("\n=== out_entropy (输出层) ===")
    for nm in res:
        print(f"{nm:24s} out_entropy={res[nm]['out_ent']:.3f}")
    print("\n=== hid_norm 轨迹 (残差流: 输入/层1/层2/层3) ===")
    for nm in res:
        print(f"{nm:24s} " + " ".join(f"{x:.3f}" for x in res[nm]['hid']))
    print("\n=== attn_entropy 轨迹 (bits) ===")
    for nm in res:
        print(f"{nm:24s} " + " ".join(f"{x:.3f}" for x in res[nm]['attn']))

    base_hid = res['untrained_s0']['hid']
    base_attn = res['untrained_s0']['attn']
    print("\n=== hid_norm 轨迹 与 untrained_s0 的皮尔逊相关 ===")
    for nm in res:
        print(f"{nm:24s} corr={corr(base_hid, res[nm]['hid']):.3f}")
    print("\n=== attn_entropy 轨迹 与 untrained_s0 的皮尔逊相关 ===")
    for nm in res:
        print(f"{nm:24s} corr={corr(base_attn, res[nm]['attn']):.3f}")

    # 关键对: 训前 vs 训后 (同一 seed)
    print("\n=== 关键比对: 同 seed 训前 vs 训后 ===")
    print(f"  hid_norm  corr = {corr(res['trained_from_s0_pretrain']['hid'], res['trained_s0']['hid']):.3f}")
    print(f"  attn_ent  corr = {corr(res['trained_from_s0_pretrain']['attn'], res['trained_s0']['attn']):.3f}")

    out = {"vocab": V, "loss_start": losses[0], "loss_end": losses[-1],
           "results": {k: {"out_ent": v["out_ent"], "hid": v["hid"], "attn": v["attn"]} for k, v in res.items()}}
    with open("xinhua_exp_numpy_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n结果写入 xinhua_exp_numpy_result.json")


if __name__ == "__main__":
    main()
