"""
Offline Evidence Quality Diagnostics (Step 1)
No pipeline rerun needed — compares result.json vs public_dataset.json.

Metrics per sample:
  - rouge_l        : ROUGE-L F against ground truth evidence
  - pred_len       : predicted evidence char length
  - gt_len         : ground truth evidence char length
  - len_ratio      : pred_len / gt_len
  - verbatim_hit   : ground truth evidence is a substring of predicted (or vice versa)
  - starts_mid_sent: predicted evidence starts mid-sentence (likely a fallback child chunk)
  - ends_mid_sent  : predicted evidence ends mid-sentence

Aggregate:
  - ROUGE-L distribution (mean/p25/p50/p75)
  - Fraction of mid-sentence starts/ends (fallback indicator)
  - Bucketed analysis: high/mid/low ROUGE-L groups
"""

import json
import re
import statistics
from pathlib import Path

from rouge_score import rouge_scorer

RESULT_PATH = "result.json"
DATASET_PATH = "public_dataset.json"

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(pred: str, golds: list[str]) -> float:
    if not golds:
        return 1.0
    if not pred:
        return 0.0
    return _rouge.score_multi(targets=golds, prediction=pred)["rougeL"].fmeasure


def starts_mid_sentence(text: str) -> bool:
    """True if text likely starts in the middle of a sentence."""
    t = text.lstrip()
    if not t:
        return False
    # Starts with lowercase letter → mid-sentence
    if t[0].islower():
        return True
    return False


def ends_mid_sentence(text: str) -> bool:
    """True if text likely ends without a sentence terminator."""
    t = text.rstrip()
    if not t:
        return False
    return t[-1] not in ".!?\"'"


def verbatim_overlap(pred: str, golds: list[str]) -> bool:
    """True if any gold evidence is substantially contained in pred, or pred in gold."""
    pred_n = re.sub(r'\s+', ' ', pred).strip().lower()
    for g in golds:
        g_n = re.sub(r'\s+', ' ', g).strip().lower()
        # Check either direction; use 80% of shorter as threshold
        shorter = min(len(pred_n), len(g_n))
        if shorter < 20:
            continue
        if g_n in pred_n or pred_n in g_n:
            return True
        # Partial: check if ≥60% of gold tokens appear in pred
        g_tokens = set(g_n.split())
        p_tokens = set(pred_n.split())
        if len(g_tokens) > 0 and len(g_tokens & p_tokens) / len(g_tokens) >= 0.6:
            return True
    return False


def main():
    results = json.loads(Path(RESULT_PATH).read_text(encoding="utf-8"))
    dataset = json.loads(Path(DATASET_PATH).read_text(encoding="utf-8"))

    # Index dataset by title
    gt_by_title = {item["title"]: item for item in dataset}

    rows = []
    missing = 0

    for r in results:
        title = r["title"]
        pred_evs = r.get("evidence", [])
        pred_ev = pred_evs[0] if pred_evs else ""

        gt = gt_by_title.get(title)
        if gt is None:
            missing += 1
            continue

        gold_evs = gt.get("evidence", [])
        rl = rouge_l(pred_ev, gold_evs)
        pred_len = len(pred_ev)
        gt_len = sum(len(g) for g in gold_evs) / max(len(gold_evs), 1)
        len_ratio = pred_len / gt_len if gt_len > 0 else 0

        rows.append({
            "title":           title,
            "rouge_l":         rl,
            "pred_len":        pred_len,
            "gt_len":          gt_len,
            "len_ratio":       len_ratio,
            "verbatim_hit":    verbatim_overlap(pred_ev, gold_evs),
            "starts_mid":      starts_mid_sentence(pred_ev),
            "ends_mid":        ends_mid_sentence(pred_ev),
            "pred_ev":         pred_ev[:120],
            "gold_ev":         gold_evs[0][:120] if gold_evs else "",
        })

    n = len(rows)
    print(f"\n{'='*60}")
    print(f"  Evidence Diagnostics  (n={n}, missing={missing})")
    print(f"{'='*60}")

    rls = [r["rouge_l"] for r in rows]
    print(f"\n── ROUGE-L Distribution ──")
    print(f"  mean : {statistics.mean(rls):.4f}")
    print(f"  stdev: {statistics.stdev(rls):.4f}")
    sorted_rls = sorted(rls)
    def p(q): return sorted_rls[int(q * n)]
    print(f"  p25  : {p(0.25):.4f}")
    print(f"  p50  : {p(0.50):.4f}")
    print(f"  p75  : {p(0.75):.4f}")
    print(f"  p90  : {p(0.90):.4f}")
    print(
        f"  ==0  : {sum(1 for x in rls if x == 0)}/{n} ({100*sum(1 for x in rls if x == 0)/n:.1f}%)")
    print(
        f"  <0.1 : {sum(1 for x in rls if x < 0.1)}/{n} ({100*sum(1 for x in rls if x < 0.1)/n:.1f}%)")
    print(
        f"  ≥0.5 : {sum(1 for x in rls if x >= 0.5)}/{n} ({100*sum(1 for x in rls if x >= 0.5)/n:.1f}%)")

    starts_mid_count = sum(1 for r in rows if r["starts_mid"])
    ends_mid_count = sum(1 for r in rows if r["ends_mid"])
    both_mid_count = sum(1 for r in rows if r["starts_mid"] and r["ends_mid"])
    verbatim_count = sum(1 for r in rows if r["verbatim_hit"])

    print(f"\n── Fallback Indicators (mid-sentence boundary) ──")
    print(
        f"  starts mid-sentence : {starts_mid_count}/{n} ({100*starts_mid_count/n:.1f}%)  ← likely child chunk fallback")
    print(
        f"  ends mid-sentence   : {ends_mid_count}/{n}  ({100*ends_mid_count/n:.1f}%)")
    print(
        f"  both mid            : {both_mid_count}/{n}  ({100*both_mid_count/n:.1f}%)")

    print(f"\n── Content Match ──")
    print(
        f"  verbatim/60%+ token overlap : {verbatim_count}/{n} ({100*verbatim_count/n:.1f}%)")
    no_verbatim_rls = [r["rouge_l"] for r in rows if not r["verbatim_hit"]]
    verbatim_rls = [r["rouge_l"] for r in rows if r["verbatim_hit"]]
    if verbatim_rls:
        print(
            f"  ROUGE-L when verbatim=True  : {statistics.mean(verbatim_rls):.4f}")
    if no_verbatim_rls:
        print(
            f"  ROUGE-L when verbatim=False : {statistics.mean(no_verbatim_rls):.4f}")

    print(f"\n── Length Analysis ──")
    ratios = [r["len_ratio"] for r in rows]
    print(f"  pred/gt len ratio  mean: {statistics.mean(ratios):.2f}")
    print(f"  pred/gt len ratio  p50 : {sorted(ratios)[n//2]:.2f}")
    too_long = sum(1 for x in ratios if x > 2.5)
    too_short = sum(1 for x in ratios if x < 0.4)
    print(f"  pred >> gt (>2.5x) : {too_long}/{n}  ({100*too_long/n:.1f}%)")
    print(f"  pred << gt (<0.4x) : {too_short}/{n} ({100*too_short/n:.1f}%)")

    # Bucket analysis
    print(f"\n── Bucketed ROUGE-L ──")
    low = [r for r in rows if r["rouge_l"] < 0.2]
    mid = [r for r in rows if 0.2 <= r["rouge_l"] < 0.5]
    high = [r for r in rows if r["rouge_l"] >= 0.5]

    for label, bucket in [("LOW (<0.2)", low), ("MID (0.2-0.5)", mid), ("HIGH (≥0.5)", high)]:
        if not bucket:
            continue
        sm = sum(1 for r in bucket if r["starts_mid"])
        em = sum(1 for r in bucket if r["ends_mid"])
        vh = sum(1 for r in bucket if r["verbatim_hit"])
        lr = statistics.mean(r["len_ratio"] for r in bucket)
        print(f"\n  {label}  n={len(bucket)}")
        print(
            f"    starts_mid={sm} ({100*sm/len(bucket):.0f}%)  ends_mid={em} ({100*em/len(bucket):.0f}%)")
        print(
            f"    verbatim_hit={vh} ({100*vh/len(bucket):.0f}%)  avg_len_ratio={lr:.2f}")

    # Worst 10
    print(f"\n── Worst 10 by ROUGE-L ──")
    worst = sorted(rows, key=lambda r: r["rouge_l"])[:10]
    for r in worst:
        print(
            f"\n  [{r['rouge_l']:.3f}] starts_mid={r['starts_mid']} ends_mid={r['ends_mid']} len_ratio={r['len_ratio']:.1f}")
        print(f"  TITLE: {r['title'][:70]}")
        print(f"  PRED : {r['pred_ev']}")
        print(f"  GOLD : {r['gold_ev']}")

    print(f"\n{'='*60}")
    print("Diagnosis summary:")
    fallback_rate = (starts_mid_count + ends_mid_count) / (2 * n)
    if starts_mid_count / n > 0.20:
        print(
            f"  ⚠ HIGH fallback rate ({100*starts_mid_count/n:.0f}% start mid-sentence) → fix LLM output format")
    else:
        print(
            f"  ✓ Low fallback rate ({100*starts_mid_count/n:.0f}% start mid-sentence)")
    if statistics.mean(ratios) > 2.0:
        print(
            f"  ⚠ Predictions too LONG (avg {statistics.mean(ratios):.1f}x ground truth) → trim evidence to 1 sentence")
    elif statistics.mean(ratios) < 0.5:
        print(f"  ⚠ Predictions too SHORT → may be truncated or wrong chunk")
    no_verbatim_frac = len(no_verbatim_rls) / n
    if no_verbatim_frac > 0.4:
        print(
            f"  ⚠ {100*no_verbatim_frac:.0f}% cases have no verbatim overlap → wrong chunk retrieved")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
