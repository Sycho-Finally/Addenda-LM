import os, json, math, gc
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig

# 4B 实机测试 (master 指定, 逻辑验证足够). 硬件: RTX 3060 6GB / RAM 16GB(空闲~2.7G).
# 两模型均 4-bit(~2.3GB) 进显存:
#   训练过   = from_pretrained(load_in_4bit)  —— safetensors mmap 流式量化, 低内存
#   随机初始化 = meta 设备建模 -> 逐层 Linear4bit -> GPU 上逐张量生成随机权重并即时量化
#              (transformers 5.17 移除了 quantize_model; 全 fp16 建模需 8GB RAM 会爆,
#               meta 流峰值内存只有单个张量, 模型直接落在显存 ~2.5GB)
# 顺序加载: 先训后随, 中间 del+gc+empty_cache.
# 只做前向, 提取"架构层走向": 逐层 hidden-state L2 范数轨迹 + 逐层 attention 熵轨迹.
# 输出层熵只打印、不进相关性: lm-head+softmax 恰是训练直接优化的对象,
# "训练过 vs 未训练"在那里必然天差地别(均匀瞎猜 vs 押中), 那是"有没有被训练过"的定义本身,
# 不含架构信息. 真正测架构的是内部动力学(残差流范数轨迹/注意力熵轨迹).
_REPO = "Qwen/Qwen3-4B-Instruct-2507"          # 不带 -2507 的旧 ID 在镜像上 404
_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "model_cache", "Qwen3-4B-Instruct-2507")
MODEL = _LOCAL if os.path.exists(os.path.join(_LOCAL, "config.json")) else _REPO
PROMPT = "猫是一种动物，它通常"
CDTYPE = torch.bfloat16
QTYPE = "nf4"


def bnb_config():
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=CDTYPE,
                              bnb_4bit_quant_type=QTYPE)


def load_trained():
    return AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config(),
                                                device_map="auto",
                                                attn_implementation="eager")


def load_random_4bit(cfg):
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit
    from bitsandbytes.functional import quantize_4bit
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        rand = AutoModelForCausalLM.from_config(cfg)      # 0 内存
    std = float(getattr(cfg, "initializer_range", 0.02))
    # 1) 换层: 除 lm_head(权重与 embedding 绑定)外的所有 Linear -> Linear4bit
    for mod_name, mod in list(rand.named_modules()):
        for ch_name, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear) and "lm_head" not in ch_name:
                setattr(mod, ch_name, Linear4bit(child.in_features, child.out_features,
                                                 bias=child.bias is not None,
                                                 compute_dtype=CDTYPE, quant_type=QTYPE))
    # 2) 逐模块赋真实权重(都在 GPU 上现场生成/量化, 峰值=单张量)
    dev = torch.device("cuda")
    for mod in rand.modules():
        tn = type(mod).__name__
        if isinstance(mod, Linear4bit):
            w = torch.empty(mod.out_features, mod.in_features, dtype=CDTYPE, device=dev).normal_(0, std)
            packed, state = quantize_4bit(w, quant_type=QTYPE, compress_statistics=True)
            # Params4bit 本身就是 nn.Parameter 子类, 直接赋值(不要再用 nn.Parameter 包装)
            mod.weight = bnb.nn.Params4bit(packed, quant_state=state, quant_type=QTYPE,
                                           compress_statistics=True, requires_grad=False)
            mod.compute_dtype = CDTYPE; mod.quant_type = QTYPE
            if mod.bias is not None:
                mod.bias = torch.nn.Parameter(torch.zeros(mod.bias.shape, dtype=CDTYPE, device=dev))
        elif isinstance(mod, torch.nn.Embedding):
            mod.weight = torch.nn.Parameter(torch.empty(
                mod.num_embeddings, mod.embedding_dim, dtype=CDTYPE, device=dev).normal_(0, std))
        elif tn.endswith("Norm") and getattr(mod, "weight", None) is not None:
            mod.weight = torch.nn.Parameter(torch.ones(mod.weight.shape[0], dtype=CDTYPE, device=dev))
    rand.tie_weights()
    # rotary 的 inv_freq 是 buffer(非 parameter), meta 流没覆盖 -> 用真类重建, 值从 config 正确计算
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        rand.model.rotary_emb = Qwen3RotaryEmbedding(config=cfg)
    except ImportError:
        pass
    rand.eval()
    return rand


def entropy_bits(logits):
    p = torch.softmax(logits.float(), dim=-1)
    return -(p * (p + 1e-12).log()).sum().item() / math.log(2)


def analyze(model, tok):
    device = next(model.parameters()).device
    inputs = tok(PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, output_attentions=True)
    out_ent = entropy_bits(out.logits[0, -1])
    hid_norms = [float(h[0].norm().item()) for h in out.hidden_states]
    attn_ent = []
    for layer_attn in out.attentions:            # (B, H, S, S)
        a = layer_attn[0].float()
        p = torch.softmax(a, dim=-1)
        e = -(p * (p + 1e-12).log()).sum(-1) / math.log(2)   # (H, S)
        attn_ent.append(float(e.mean().item()))
    return {"out_entropy_bits": out_ent, "hid_norm_traj": hid_norms, "attn_ent_traj": attn_ent}


def corr(a, b):
    a = np.array(a); b = np.array(b)
    if a.std() < 1e-9 or b.std() < 1e-9 or a.shape != b.shape:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def trend(x):
    return x[-1] / (x[0] + 1e-9)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg = AutoConfig.from_pretrained(MODEL)
    V = cfg.vocab_size

    print(f"=== 加载 训练过 模型 ({MODEL}, 4bit, device_map=auto) ===", flush=True)
    trained = load_trained()
    r_t = analyze(trained, tok)
    del trained; gc.collect(); torch.cuda.empty_cache()

    print(f"=== 加载 随机初始化 模型 (meta流+逐层4bit量化) ===", flush=True)
    rand = load_random_4bit(cfg)
    r_r = analyze(rand, tok)
    del rand; gc.collect(); torch.cuda.empty_cache()

    print("\n=== 架构参数 ===")
    print(f"model={MODEL} layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} vocab={V}")
    print(f"随机输出熵理论上限 ~= {math.log2(V):.2f} bits")

    print("\n=== 输出熵 (bits; 只打印不进相关性: 见文件头注释) ===")
    print(f"trained : {r_t['out_entropy_bits']:.3f}")
    print(f"random  : {r_r['out_entropy_bits']:.3f}")

    print("\n=== 逐层 hidden-state L2 范数轨迹 (残差流走向) ===")
    for i, (a, b) in enumerate(zip(r_t['hid_norm_traj'], r_r['hid_norm_traj'])):
        print(f"L{i:>3} : {a:10.3f} | {b:10.3f}")

    print("\n=== 逐层 attention 熵轨迹 (bits, 越低越集中) ===")
    for i, (a, b) in enumerate(zip(r_t['attn_ent_traj'], r_r['attn_ent_traj'])):
        print(f"L{i:>3} : {a:10.3f} | {b:10.3f}")

    print("\n=== 一致性度量 (架构层走向是否同构) ===")
    hc = corr(r_t['hid_norm_traj'], r_r['hid_norm_traj'])
    ac = corr(r_t['attn_ent_traj'], r_r['attn_ent_traj'])
    print(f"hid_norm 轨迹皮尔逊相关 : {hc:.3f}")
    print(f"hid_norm 趋势(末/首)    : trained={trend(r_t['hid_norm_traj']):.3f} "
          f"random={trend(r_r['hid_norm_traj']):.3f}")
    print(f"attn_ent 轨迹皮尔逊相关 : {ac:.3f}")

    res = {"model": str(MODEL), "vocab": V, "quant": "4bit-nf4",
           "trained": r_t, "random": r_r, "hid_corr": hc, "attn_corr": ac,
           "hid_trend_trained": trend(r_t['hid_norm_traj']),
           "hid_trend_random": trend(r_r['hid_norm_traj'])}
    with open("xinhua_exp_result.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\n结果已写入 xinhua_exp_result.json")


if __name__ == "__main__":
    main()
