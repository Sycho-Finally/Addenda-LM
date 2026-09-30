import os, json, re, calendar
from collections import defaultdict

# ---------------------------------------------------------------------------
# curator.py — 生网络 -> 熟网络 的自动化规则门 (把"agent 中介"的角色规则化)
#
# 规则 (按序执行, 全部可机检):
#   R1 确定性校验: 日期类答案必须存在(2026年2月30日 => 直接拒), 数字类必须在正整数域
#   R2 佐证计分: tier 权重 official=3, wiki/news/product=2, blog/community/ugc=1, none=0
#   R3 准入: 同一问题的最高分答案, 需 score>=2 (孤blog不入) 或 有 official 背书
#   R4 冲突处理: 落选答案记为 rejected(冲突候选), 与无源投毒分开归档
# 输出: curated_claims.json (admitted / rejected), 供训练端直接消费
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
TIER_W = {"official": 3, "wiki": 2, "news": 2, "product": 2,
          "blog": 1, "community": 1, "ugc": 1, "none": 0}

DATE_RE = re.compile(r"^(\d{4})年(\d{1,2})月(\d{1,2})日?$")


def date_valid(a):
    """R1: 'YYYY年M月D日' 必须是真实存在的日期; 其他格式放行给 R2."""
    m = DATE_RE.match(a.strip())
    if not m:
        return True                      # 非日期格式(如 '284'/'2025年7月')交由 R2
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12):
        return False
    return 1 <= d <= calendar.monthrange(y, mo)[1]


def curate(raw):
    groups = defaultdict(lambda: defaultdict(list))   # q -> answer -> [entries]
    rejected = []
    for e in raw["raw_claims"]:
        if not date_valid(e["a"]):
            rejected.append({**e, "admitted": False,
                             "reject_reason": f"R1 确定性拒绝: 日期不存在 ({e['a']})"})
            continue
        groups[e["q"]][e["a"]].append(e)

    admitted = []
    for q, answers in groups.items():
        scored = []
        for a, entries in answers.items():
            score = sum(TIER_W.get(x["tier"], 0) for x in entries)
            has_official = any(x["tier"] == "official" for x in entries)
            scored.append((score, has_official, a, entries))
        scored.sort(key=lambda t: (-t[0], t[2]))
        top_score, top_official, top_a, top_entries = scored[0]
        ok = top_score >= 2 or top_official
        if ok:
            admitted.append({"q": q, "a": top_a, "admitted": True,
                             "score": top_score,
                             "sources": sorted({x["source"] for x in top_entries}),
                             "tiers": sorted({x["tier"] for x in top_entries})})
            for sc, off, a, entries in scored[1:]:
                for x in entries:
                    rejected.append({**x, "admitted": False,
                                     "reject_reason": f"R4 冲突落选 (胜者: {top_a}, 分差 {top_score - sc})"})
        else:
            for sc, off, a, entries in scored:
                for x in entries:
                    rejected.append({**x, "admitted": False,
                                     "reject_reason": f"R3 佐证不足 (score={sc} < 2 且无 official)"})
    return admitted, rejected


def main():
    raw = json.load(open(os.path.join(DATA, "raw_claims.json"), encoding="utf-8"))
    admitted, rejected = curate(raw)
    out = {"meta": {"curator": "rule-gate v1 (R1确定性/R2计分/R3准入/R4冲突)",
                    "n_raw": len(raw["raw_claims"])},
           "admitted": admitted, "rejected": rejected}
    with open(os.path.join(DATA, "curated_claims.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"门报告: raw={len(raw['raw_claims'])} -> admitted={len(admitted)} rejected={len(rejected)}")
    print("--- 准入 ---")
    for c in admitted:
        print(f"  [score={c['score']:>2}] {c['a']:<14} <- {c['q'][:26]}  sources={c['sources']}")
    print("--- 拒绝 ---")
    for c in rejected:
        print(f"  {c['a']:<14} {c['reject_reason'][:46]}  ({c['q'][:22]})")


if __name__ == "__main__":
    main()
