import os, json, math, gc, hashlib
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# 模块图谱 · 不遗忘实验 (最低成本版)
# 问题: "旧参数永不被改 => 旧能力不被破坏" 目前只是定义推演. 本实验把它变成数据.
# 设计:
#   - 冻结 Qwen3-4B-Instruct-2507 (4bit), 主干前向全程 no_grad, 取末层隐状态后 detach
#   - 挂一个 additive 残差分支: h' = h + MEM(h), 1.3M 参数, 末层零初始化
#     (零初始化 => 训练起点 h'==h, 行为与原模型逐位一致, 任务A不可能在起点就掉)
#   - 任务A: 20 个稳定常识问答 (测保持)  任务B: 16 个编造的"名字->代号"事实 (测可学)
#   - 只训 MEM 学任务B; 前后两次: 任务A gold-answer logprob + 任务B teacher-forced 准确率
#   - 主干权重采样 SHA256 前后对比: 证明物理上没被碰
# 判定: B 大幅上升 且 A 不动 => "不遗忘"从定义变成数据.
# 范围声明: 只测行为层"additive 不遗忘", 不测路由/检索/多模块/闸门.
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "model_cache", "Qwen3-4B-Instruct-2507")

TASK_A = [  # (问题, 金标答案)
    ("水的化学式是什么？只用最简式回答。", "H2O"),
    ("中国的首都是哪座城市？", "北京"),
    ("一年有多少个月？", "12"),
    ("太阳从哪个方向升起？", "东方"),
    ("世界上最大的海洋是哪个？", "太平洋"),
    ("《西游记》的作者是谁？", "吴承恩"),
    ("圆周率小数点后第一位是多少？", "1"),
    ("地球绕太阳公转一圈需要多长时间？", "一年"),
    ("人体正常体温大约是多少摄氏度？用整数回答。", "37"),
    ("珠穆朗玛峰位于哪个山脉？", "喜马拉雅山脉"),
    ("铁的元素符号是什么？", "Fe"),
    ("一打等于多少个？", "12"),
    ("企鹅主要生活在哪个大洲？", "南极洲"),
    ("正方形有几条边？", "4"),
    ("春节是农历的哪一天？", "正月初一"),
    ("空气中含量最多的气体是什么？", "氮气"),
    ("十二生肖排在第一位的动物是什么？", "鼠"),
    ("黄河最终注入哪个海？", "渤海"),
    ("一小时有多少分钟？", "60"),
    ("光在真空中的传播速度比声音快还是慢？", "快"),
]

B_NAMES = ["墨渊", "青梧", "白泽", "流萤", "孤鹜", "寒潭", "赤羽", "素锦",
           "听澜", "折柳", "惊鸿", "落雁", "栖梧", "扶摇", "望舒", "既明"]
B_CODES = ["琥珀", "松柏", "蝉鸣", "霜降", "萤火", "孤帆", "远山", "长风",
           "皓月", "疏影", "浮萍", "惊蛰", "白鹭", "流云", "暮雪", "晨钟"]
TASK_B = list(zip(B_NAMES, B_CODES))
B_Q = "{}的专属代号是什么？只用一个词回答。"


def hash_base(model, n=40):
    """采样主干权重做 SHA256 (证明训练前后物理未变)."""
    h = hashlib.sha256()
    sd = model.state_dict()
    names = sorted(sd.keys())
    step = max(1, len(names) // n)
    for k in names[::step]:
        t = sd[k]
        try:
            raw = t.detach().cpu().contiguous()
            if raw.dtype != torch.uint8:
                raw = raw.view(torch.uint8)          # 按字节重解释, 与 dtype 无关
            b = raw.numpy().tobytes()
        except Exception:
            b = str(tuple(t.shape)).encode()
        h.update(k.encode()); h.update(b)
    return h.hexdigest()[:16]


def build_example(tok, question, answer):
    """chat template 拼接, 返回 (ids, labels), labels 只在答案 token 上."""
    prompt_text = tok.apply_chat_template([{"role": "user", "content": question}],
                                          add_generation_prompt=True, tokenize=False)
    prompt = tok(prompt_text, add_special_tokens=False).input_ids
    ans_ids = tok(answer, add_special_tokens=False).input_ids
    ans_ids = ans_ids + [tok.convert_tokens_to_ids("<|im_end|>")]
    ids = prompt + ans_ids
    labels = [-100] * len(prompt) + ans_ids
    return ids, labels


@torch.no_grad()
def eval_task(tok, model, data, mem=None, bs=4):
    """mean gold-answer logprob per item (越高说明对该答案越确信)."""
    model.eval()
    scores = []
    for i in range(0, len(data), bs):
        chunk = data[i:i + bs]
        exs = [build_example(tok, q, a) for q, a in chunk]
        L = max(len(x[0]) for x in exs)
        ids = torch.full((len(exs), L), tok.pad_token_id or 0, dtype=torch.long)
        lab = torch.full((len(exs), L), -100, dtype=torch.long)
        msk = torch.zeros((len(exs), L), dtype=torch.long)
        for j, (x, y) in enumerate(exs):
            ids[j, :len(x)] = torch.tensor(x); lab[j, :len(y)] = torch.tensor(y)
            msk[j, :len(x)] = 1
        ids, lab, msk = ids.cuda(), lab.cuda(), msk.cuda()
        out = model.model(input_ids=ids, attention_mask=msk, output_hidden_states=True)
        h = out.hidden_states[-1].detach()          # 主干冻结, 图从这里才开始
        logits = model.lm_head(model.model.norm(mem(h) if mem is not None else h))
        logp = torch.log_softmax(logits.float(), -1)
        gl = torch.gather(logp[:, :-1], 2, lab[:, 1:].clamp(min=0).unsqueeze(-1)).squeeze(-1)
        cnt = (lab[:, 1:] != -100).sum(-1).clamp(min=1)
        scores += (gl * (lab[:, 1:] != -100)).sum(-1) / cnt
    return [float(s) for s in scores]


class MemBranch(nn.Module):
    """additive 残差分支: h' = h + W2·gelu(W1·h). 末层零初始化 => 起点恒等."""
    def __init__(self, d=2560, r=256):
        super().__init__()
        self.w1 = nn.Linear(d, r, dtype=torch.float32)
        self.w2 = nn.Linear(r, d, dtype=torch.float32)
        nn.init.normal_(self.w1.weight, 0, 0.02); nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.weight); nn.init.zeros_(self.w2.bias)
    def forward(self, h):                        # h: (B,T,d) bf16
        return h + self.w2(nn.functional.gelu(self.w1(h.float()))).to(h.dtype)


def train_mem(tok, model, mem, steps=240, bs=4, lr=1e-3):
    opt = torch.optim.AdamW(mem.parameters(), lr=lr)
    exs = [build_example(tok, B_Q.format(n), c) for n, c in TASK_B]
    rng = np.random.default_rng(0)
    model.eval()
    for st in range(steps):
        idx = rng.choice(len(exs), bs, replace=False)
        chunk = [exs[i] for i in idx]
        L = max(len(x[0]) for x in chunk)
        ids = torch.full((bs, L), tok.pad_token_id or 0, dtype=torch.long)
        lab = torch.full((bs, L), -100, dtype=torch.long)
        msk = torch.zeros((bs, L), dtype=torch.long)
        for j, (x, y) in enumerate(chunk):
            ids[j, :len(x)] = torch.tensor(x); lab[j, :len(y)] = torch.tensor(y)
            msk[j, :len(x)] = 1
        ids, lab, msk = ids.cuda(), lab.cuda(), msk.cuda()
        with torch.no_grad():
            out = model.model(input_ids=ids, attention_mask=msk, output_hidden_states=True)
            h = out.hidden_states[-1].detach()
        logits = model.lm_head(model.model.norm(mem(h)))     # 图: 只经过 mem/norm/head
        gl = logits[:, :-1]
        loss = nn.functional.cross_entropy(
            gl.reshape(-1, gl.shape[-1]).float(), lab[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 40 == 0:
            print(f"  step {st:3d} loss={loss.item():.4f}", flush=True)
    return float(loss.item())


@torch.no_grad()
def b_accuracy(tok, model, mem):
    """任务B teacher-forced: 答案首 token argmax 命中率."""
    model.eval(); mem.eval()
    hit = 0
    for n, c in TASK_B:
        ids, _ = build_example(tok, B_Q.format(n), c)
        t = torch.tensor(ids)[:-1].unsqueeze(0).cuda()
        with torch.no_grad():
            out = model.model(input_ids=t, output_hidden_states=True)
            h = out.hidden_states[-1].detach()
        logits = model.lm_head(model.model.norm(mem(h)))
        pos = len(ids) - len(tok(c, add_special_tokens=False).input_ids) - 2  # 首答案 token 位置
        pred = logits[0, pos].argmax().item()
        gold = tok(c, add_special_tokens=False).input_ids[0]
        hit += int(pred == gold)
    return hit / len(TASK_B)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=__import__('transformers', fromlist=['BitsAndBytesConfig']).BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"),
        device_map="auto", attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.hidden_size
    mem = MemBranch(d=d).cuda()
    n_par = sum(p.numel() for p in mem.parameters())
    h0 = hash_base(model)
    print(f"base={d}d mem_params={n_par} base_hash={h0}", flush=True)

    a0 = eval_task(tok, model, TASK_A, mem=mem)  # mem 零初始化=恒等, 与纯主干逐位一致
    b0_acc = b_accuracy(tok, model, mem)
    print(f"[baseline] A mean_logp={np.mean(a0):.4f}  B_acc={b0_acc:.3f}", flush=True)

    print("=== 训练 MEM (任务B only, 主干冻结) ===", flush=True)
    train_mem(tok, model, mem)

    h1 = hash_base(model)
    a1 = eval_task(tok, model, TASK_A, mem=mem)  # 关键口径: 挂上训过的 mem 再测 A
    b1_acc = b_accuracy(tok, model, mem)
    da = [b - a for b, a in zip(a1, a0)]
    print(f"[after]     A mean_logp={np.mean(a1):.4f}  B_acc={b1_acc:.3f}", flush=True)
    print(f"base_hash {'UNCHANGED' if h1 == h0 else 'CHANGED!'} ({h1})", flush=True)
    print(f"A delta: mean={np.mean(da):+.4f}  max|d|={np.max(np.abs(da)):.4f}")
    flips = sum(1 for x in da if x < -0.3)
    print(f"A items with logp drop > 0.3: {flips}/{len(TASK_A)}")
    for (q, _), x in zip(TASK_A, da):
        if abs(x) > 0.15:
            print(f"   |d|={x:+.3f}  {q[:18]}")

    res = {"A_mean_logp_before": float(np.mean(a0)), "A_mean_logp_after": float(np.mean(a1)),
           "A_delta_mean": float(np.mean(da)), "A_delta_maxabs": float(np.max(np.abs(da))),
           "A_items": [float(x) for x in a0], "A_items_after": [float(x) for x in a1],
           "B_acc_before": b0_acc, "B_acc_after": b1_acc,
           "base_hash_before": h0, "base_hash_after": h1,
           "mem_params": n_par, "verdict":
               "NO-FORGETTING CONFIRMED" if (h1 == h0 and abs(np.mean(da)) < 0.05 and b1_acc > b0_acc + 0.3)
               else "CHECK DETAILS"}
    with open("mem_continual_result.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("结果写入 mem_continual_result.json", flush=True)


if __name__ == "__main__":
    main()
