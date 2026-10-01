# Addenda-LM ｜ 补遗：冻结基座上的模块化持续学习 · 全链路实验验证

**一句话**：把大语言模型当作一本**已经印刷定稿的书**——不重印、不改字，新知识以**补遗（Addenda）**的形式增补：内容门核证、行为锚定、路由隔离、技能模块跨层注入。全部流程在一个本地部署的 4B 模型上做了**初步验证**：**6GB 消费级显卡**、单实验 1.5~8 分钟、脚本与结果随仓库附带（本机验证可复现）。

> 作者：Sycho-Finally（独立研究者）｜ AI 使用声明见 [AI_DISCLOSURE.md](AI_DISCLOSURE.md) ｜ License: MIT

---

## 这个系统能做什么（四件事，均有实验数字）

1. **教模型新知识，不用重新训练基座**——16 条新事实 1 分钟学会（0%→100%），20 道旧问题在测试集上 0 退化，主干权重哈希前后一致（逐参数未触碰）。
2. **教它"手艺"，不只是"知识"**——一张从不在 prompt 出现的随机密码表，训练后模型能对**从未见过的数字**逐位编码（unseen exact 0→**1.000**）；该结果支持"学到的是程序而非死记"的解读，边界见「已知局限」。
3. **它按检索结果选择模块，未路由输入的输出与原模型逐位一致（实测）**——路由决策零错误；对照：全局激活会使旧任务漂移 -13.8（16/20 退化）。
4. **有看门的：过滤什么配被学会**——多源佐证/官方源/确定性校验三规则的内容门，17 条真实网络声明（含真实冲突与投毒）全部正确裁决；投毒与真知在无门时**同速固化**，准入判定必须在权重之外。

## 结果速览

**架构稳定容器**：trained vs random-init 残差流走向皮尔逊 **0.814**（36 层逐层，Qwen3-4B 实机）。

**不遗忘 + 相变曲线**（跨层联合分支，unseen exact / 逐位，随机水平 0.10）：

| 训练样本 | 16 | 64 | 128 | 256 |
|---|---|---|---|---|
| 密码任务 unseen exact | 0.000 | 0.375 | 0.938 | **1.000** |
| unseen 逐位 | 0.122 | 0.722 | 0.865 | **0.973** |

**漂移三分解**（全局激活，20 条旧任务）：执行噪声（bf16，数学恒等的模块也产生）**-2.17 / 80.2%**；真实语义效应仅 +0.53。→ 测量全局激活必须三分解，否则测的是噪声。

**对照（无锚定/无路由/无门）**：真事实训练致旧任务 -7.455（14/20）；投毒与真知同速固化；全局激活旧任务 -13.8（16/20）。

完整分析见 [reports/实验总结报告.md](reports/实验总结报告.md)。

## 这个仓库不是什么

先说边界，免得浪费你的时间，也免得我被打脸：

- **不是"解决了灾难性遗忘"**。验证的是：冻结基座 + 附加模块在本文条件下（Qwen3-4B、6GB、特定任务族），旧能力可以逐位不变、新知识可以注入。通用持续学习未解决，业界也未解决。
- **不是新 SOTA 方法**。全部组件思路来自公开研究（见「引用的先行者」），本仓库的增量是：全链路闭环验证、投毒门、bf16 执行噪声的三分解测量、记忆/程序相变曲线。
- **不是大规模评估**。单模型（Qwen3-4B）、单 GPU、多为单种子、任务为玩具级探针 + 少量真实核证事实；没有实跑 LoRA/O-LoRA 等标准基线对照（文中引用的基线数字系转述文献）；多 seed、第二基座、标准 CL 基准协议在 TODO 清单。
- **不是可直接上生产的系统**。采集层（自动从交互流获取知识）未实现——知识写入仅在人工供给并核证后发生；全局激活存在 bf16 噪声地板；门是演示级规则门，不是完备的事实核查系统。
- **不是该方向的首个方案或综述**。2026 年该方向研究极度活跃（见下方引用清单新增条目：Brainstacks、Engram Adapter、Markov Matrix、DMoE、PaST 等），本仓库定位是**独立验证与边界测绘**：在消费级硬件上复核"冻结基座+附加模块"模式的关键性质，并记录测量陷阱。
- **不是"有长期维护承诺的产品"**。个人研究项目，业余时间维护——issue 与讨论的响应可能很慢甚至不响应；`results/` 内是 2026-09-30 的研究快照，不构成持续更新承诺。

反过来说，它**是**什么：一个每张表都能在 6GB 显卡上 10 分钟内复现的最小闭环，和一份把"哪条结论有多硬"写清楚的实验笔记。

## 使用流程（人机分工）

```
① 决定教什么          ← 人（方向与裁决，永远留给人）
② 供给知识源          ← 人指挥 agent 检索/整理（未来：采集层自动化）
③ 核证（过门）        ← curator.py 规则门 + 人最终确认
④ 训练（怎么教）      ← 机器：梯度下降写入模块（模块已挂在模型上，零初始化=不存在）
⑤ 验收（回归测试）    ← 机器跑旧任务回归 + 人看结果
```

关键性质：**被教的是模块，不是模型**。主干 36 层全程冻结（哈希作证）；教错了摘模块，模型无损。**架构不会自动获取知识源**——知识写入仅发生在人主动"教"时，推理过程纯只读；采集层自动化（交互流→门）是设计中的下一部件，且必须与自动回归同时上线。

---

## 环境要求

- NVIDIA GPU，显存 ≥ 6GB（4bit 量化）；仅推理，无需训练基座
- Python 3.10+，Windows/Linux 均可
- 磁盘 ≥ 10GB（模型权重 8.04GB）

### 安装

```bash
# ① PyTorch（注意：PyPI 默认给 CPU 版，4bit 量化必须 CUDA 版，从官方源装）
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# ② 其余依赖
pip install transformers accelerate bitsandbytes safetensors tokenizers huggingface_hub numpy requests
```

> ⚠️ **已知的坑**（都踩过，写在这里省你半天）：
> - `pip install torch` 默认装 **CPU 版**（`+cpu`），bitsandbytes 4bit 无法工作，必须从 pytorch 官方源装 CUDA 版
> - 若装过与 torch 版本不匹配的 **torchvision**，会报 `operator torchvision::nms does not exist` 并连带炸掉 transformers——卸载 torchvision/torchaudio 即可
> - transformers 5.x 移除了 `quantize_model`；bitsandbytes 的 `Params4bit` 本身是 Parameter 子类，直接赋值不要再用 `nn.Parameter()` 包装
> - HF 下载走镜像时加 `HF_ENDPOINT=https://hf-mirror.com` 和 `HF_HUB_DISABLE_XET=1`（镜像无法代理 Xet 存储，否则 401）

### 下载模型（8.04GB，支持断点续传）

```bash
python experiments/download_model.py
# 等价于 snapshot_download('Qwen/Qwen3-4B-Instruct-2507') 到 experiments/model_cache/
# 注意仓库名必须带 -2507 后缀（不带后缀的旧 ID 在镜像上 404）
```

### 复现实验

```bash
# 全部脚本从仓库根目录运行；结果 JSON 输出到运行目录
python experiments/xinhua_exp.py            # ① 走向测试 (~1 min)
python experiments/mem_continual_test.py    # ② 不遗忘 (~2 min)
python experiments/curator.py               # ③ 规则门演示 (即时)
python experiments/auto_pipeline_test.py    # ④ 端到端 (~2 min)
python experiments/routing_test.py          # ⑤ 路由 (~2 min)
python experiments/skill_test2.py           # ⑥ 密码技能-顶层 (~3 min)
python experiments/skill_mid_test.py        # ⑦ 密码技能-中层 (~4 min)
python experiments/skill_multi_test.py      # ⑧ 跨层+数据量曲线 (~8 min/点)
#   数据量可用环境变量调节: N_TRAIN=256 STEPS=800 python experiments/skill_multi_test.py
python experiments/pegp_multilayer_test.py  # ⑨ PEGP 零空间投影 (~7 min)
python experiments/diag_pegp.py             # ⑩ PEGP 诊断工具 (~2 min)
python experiments/xinhua_exp_numpy.py      # ⑪ numpy 方法学模拟 (无 GPU 要求)
```

---

## 实验清单

| 脚本 | 验证内容 | 关键结果 | 结果文件 (results/) |
|---|---|---|---|
| `xinhua_exp.py` | 架构层走向是否由架构决定 | 走向相关 0.814；bf16 ULP 平台 | `xinhua_exp_result.json` |
| `xinhua_exp_numpy.py` | numpy 方法学模拟（多种子） | 多种子走向相关 0.97-0.98 | `xinhua_exp_numpy_result.json` |
| `mem_continual_test.py` | additive 注入不遗忘 + 哈希 | B 0→100%，A 0/20，哈希不变 | `mem_continual_result.json` |
| `web_gate_test.py` | 真事实也漂移 + 投毒同速 | A -7.455 (14/20)；投毒 0→1.00 | `web_gate_result.json` |
| `web_gate_fix.py` | replay 锚压漂移 | A 漂移 +0.212（35 倍改善） | `web_gate_fix_result.json` |
| `curator.py` | 规则门自动熟化 | 17 raw → 5 准入 5 拒绝，全对 | `curated_claims.json` |
| `auto_pipeline_test.py` | raw→门→训练 端到端 | B 0→1.00，A +0.201，1/20 | `auto_pipeline_result.json` |
| `routing_test.py` | 路由结构性隔离 | A 逐位为零 (bit-exact) | `routing_result.json` |
| `skill_test2.py` | 顶层=记忆 only | unseen 0.000 | `skill_result_v2.json` |
| `skill_mid_test.py` | 中层修复记忆不修复泛化 | seen 1.000 / unseen 0.000 | （stdout） |
| `skill_multi_test.py` | 跨层×数据量相变曲线 | 256 样本 unseen 1.000 | `skill_multi_result_*.json` |
| `pegp_multilayer_test.py` | PEGP 零空间投影 | 锚输出 1e-6；噪声地板暴露 | `pegp_result.json` |
| `pegp_decomp_test.py` | 漂移三分解 | 执行噪声占 80.2% | `pegp_decomp_result.json` |
| `diag_pegp.py` | 逐层诊断工具 | 定位执行噪声根因 | （stdout） |

---

## 测量方法论（重要）

全局激活实验的漂移测量**必须做三分解**：`直通（off）`、`挂零初始化分支（zero，数学恒等但执行上下文相同）`、`挂训练后分支（trained）`。
`zero − off` 是 bf16 执行噪声（分支矩阵乘改变 cuBLAS 上下文，大激活坐标 ~7000 处 ULP=32 产生 ±32 噪声），实测占表观漂移的 **80.2%**；`trained − zero` 才是模块的真实语义效应。
不做三分解，会把机器抖动误判为模型学坏。路由（不执行分支矩阵乘）是唯一逐位精确的隔离方案。

---

## 已知局限（诚实边界）

- 技能模块需要 O(100)+ 多样样本才诱导泛化电路；事实类只需 O(10)
- 全局激活存在 bf16 噪声地板（可统计正则压语义、无法消除执行噪声）；路由是结构解
- 只验证了事实注入与单步程序（密码编码）；多步推理类技能未测
- 词法路由阈值需对背景分布校准（见 routing 实验的 0.20 边界案例）
- 单 GPU、单模型（Qwen3 系）验证；机制理论上模型无关，未实测其他家族

---

## 引用的先行者（部分思路来自这些工作）

> 该方向 2026 年研究极度活跃；以下仅列与本仓库组件直接相关者，未穷尽。

- Memory Layers at Scale (Meta FAIR, arXiv:2412.09764) — 模块容量形态
- MAC: Memory of Amortized Contexts — 冻结基座免梯度在线适应
- O-LoRA (EMNLP 2023 Findings, arXiv:2310.14152) / PEGP (arXiv:2405.13383) / GPM (ICLR 2021) — 正交零空间防遗忘
- SCALE (ACL 2026 Findings) — 冻结基座宽度扩展与可证保持
- Macaron-V1 (arXiv:2608.09819) — 冻结基座 + 专家路由的生产实践
- CL under Backdoor Attacks (arXiv:2609.06346) — 持续学习投毒防御
- GraftLLM (arXiv, SkillPack) — 模块化技能包 + 免遗忘持续学习 + 路由
- **Engram Adapter** (EMNLP 2026 Findings, arXiv:2608.29327) — 条件记忆适配器（同基座 Qwen3-4B；n-gram 选择先验 + 标量门，与本文路由组件同构）
- **Memory as a Markov Matrix** (ICML 2026, arXiv:2605.04308) — token-to-dictionary 映射、可证零遗忘与样本复杂度界（与本文"字典+补遗"设定同构的理论版）
- **Brainstacks** (arXiv:2604.01152) — 冻结 MoE-LoRA 栈 + 零空间投影 + outcome 路由
- **DMoE** (arXiv:2606.14243) — 与基座解耦的专家做参数化知识注入（末层 FFN 挂载）
- **PaST** (arXiv:2601.11258) — 知识(SFT)与技能(RL)更新的近正交性与技能向量线性注入
- **JumpLoRA** (arXiv:2604.16171) — JumpReLU 稀疏参数隔离
- **Improving Sparse Memory Finetuning** (arXiv:2604.05248) — 消费级硬件上的稀疏记忆改造 + KL 槽位选择
- **Mechanistic Analysis of Catastrophic Forgetting** (arXiv:2601.18699) — 20 个 LLM 的遗忘机制分析与分层定位

---

## English (Quick Summary)

A full-pipeline experimental validation of **modular continual learning on a frozen 4B LLM**, on consumer hardware (6GB GPU): content gate (multi-source verification, poison defense) → behavioral anchoring (replay KL, 35× drift reduction) → lexical routing (bit-exact isolation, ±0 drift on unrouted inputs) → skill modules (multi-layer joint branches; phase transition from memorization to **100% generalization** at 256 examples) → measurement methodology (3-way decomposition showing **80% of naive drift is bf16 execution noise**). All experiments reproduce in minutes; see the tables above and [reports/实验总结报告.md](reports/实验总结报告.md).

## License

MIT — 见 [LICENSE](LICENSE)。基座模型 Qwen3-4B-Instruct-2507 © Alibaba Cloud, Apache 2.0。
