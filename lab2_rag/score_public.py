"""
HW2 RAG — Self-Scoring Script (Public Dataset)

Compare your pipeline output against the public dataset ground truth.

  Evidence Score  — ROUGE-L F-measure (identical algorithm to TA grading)
  Correctness     — Lightweight LLM judge using the same 3B model you run for RAG
                    ⚠ This is an approximation: TA grading uses a separate, more
                    capable model with a different evaluation procedure.

Usage:
  python score_public.py results.json
  python score_public.py results.json --port 8091
  python score_public.py results.json --port 8091 --host 192.168.0.7

Input JSON format (same as submission):
  [{"title": "...", "answer": "...", "evidence": [...]}, ...]

Output:
  - Console: per-paper scores and running averages
  - results_score.json: full breakdown
"""

import argparse
import json
import re
import sys
from pathlib import Path

from langchain_community.llms.vllm import VLLMOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from rouge_score import rouge_scorer

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Self-score HW2 results on the public dataset")
parser.add_argument("results",  type=str,                                        help="Path to your results JSON.")
parser.add_argument("--host",   type=str, default="localhost",                   help="LLM server host. Default: localhost")
parser.add_argument("--port",   type=int, default=8091,                          help="LLM API port (same server used for RAG). Default: 8091")
parser.add_argument("--model",  type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="LLM model name.")
parser.add_argument("--dataset",type=str, default="datasets/public_dataset.json",help="Path to the public dataset JSON.")
parser.add_argument("--times",  type=int, default=5,                             help="Judge runs per paper (majority vote). Default: 5")
args = parser.parse_args()

LLM_ENDPOINT = f"http://{args.host}:{args.port}/v1"

# ──────────────────────────────────────────────────────────────────────────────
# Evidence Score — ROUGE-L (identical to TA grading)
# ──────────────────────────────────────────────────────────────────────────────
_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def calc_evidence_score(retrieved: list[str], golden: list[str]) -> float:
  if not golden:
    return 1.0
  if not retrieved:
    return 0.0
  return sum(
    _rouge.score_multi(targets=golden, prediction=c)["rougeL"].fmeasure
    for c in retrieved
  ) / len(retrieved)

# ──────────────────────────────────────────────────────────────────────────────
# LLM Judge
# ──────────────────────────────────────────────────────────────────────────────
_PROMPT_JUDGEMENT = """\
You are an expert evaluator. Given a document, a question, a ground truth answer, and a model prediction, decide if the prediction is correct.

Evaluation rules (apply in order, stop at the first match):
a) Ground Truth is always correct by definition.
b) If the Prediction expresses uncertainty (e.g. "I don't know", "I'm not sure"), score = 0.
c) If the Prediction exactly matches the Ground Truth (case-insensitive), score = 1.
d) If the Ground Truth is a number, score = 1 only if the Prediction states the same number.
e) If the Prediction is self-contradictory or does not address the question, score = 0.
f) If the Prediction is semantically equivalent to the Ground Truth or is a correct concise summary of it, score = 1.
g) If the Ground Truth lists multiple items, score = 1 only if the Prediction covers all key items (exact wording not required; semantic match is acceptable).
h) Otherwise, score = 0.

IMPORTANT: The score MUST be exactly 0 or 1. No other value is valid.
"""
_JUDGE_TEMPLATE = (
  f"system: {_PROMPT_JUDGEMENT}\n"
  "human:\n"
  "document: {document}\n"
  "question: {question}\n"
  "Ground Truth: {answer}\n"
  "Prediction: {prediction}\n"
  "assistant: The score is "
)

_llm = VLLMOpenAI(
  base_url=LLM_ENDPOINT,
  api_key="abc",
  model=args.model,
  temperature=0.6,
  max_tokens=1,
  frequency_penalty=1.6, 
  presence_penalty=0.8,
  model_kwargs={"stop": ["```", "}}"]},
)
_judge_chain = PromptTemplate.from_template(_JUDGE_TEMPLATE) | _llm | StrOutputParser()

_IDK_RE = re.compile(
  r"i (don'?t|do not|cannot|can'?t) know"
  r"|i'?m not sure"
  r"|not (enough|sufficient) (information|context|detail)"
  r"|the (context|document|passage|text) does(n'?t| not) (contain|mention|provide|have|include|say)",
  re.IGNORECASE,
)

def _extract_score(text: str) -> float:
  """Extract 'The score is X' from judge output. Fallback: final standalone digit."""
  m = re.search(r"The score is ([01])(?!\d)", text)
  if m:
    return float(m.group(1))
  m = re.search(r"([01])\s*$", text.strip())
  return float(m.group(1)) if m else 0.0

def judge_correctness(
  title: str,
  question: str,
  gt_evidence: list[str],
  gt_answer: list[str],
  prediction: str,
  times: int = 3,
) -> float:
  """Run judge `times` times and return majority-vote score (0.0 or 1.0)."""
  if not prediction.strip() or prediction in {"N/A", "n/a", "None", "Answer:"}:
    return 0.0
  if _IDK_RE.search(prediction):
    return 0.0
  query = {
    "document":   f"Paper title: {title}\n" + "\n".join(gt_evidence),
    "question":   question,
    "answer":     " ".join(gt_answer),
    "prediction": prediction,
  }
  scores: list[float] = []
  for _ in range(times):
    try:
      raw = _judge_chain.invoke(query)
      scores.append(_extract_score(raw))
    except Exception as exc:
      print(f"  [judge error: {exc}]", file=sys.stderr)
  if not scores:
    return 0.0
  return 1.0 if sum(scores) / len(scores) >= 0.5 else 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────
try:
  with open(args.dataset) as f:
    gt_map = {item["title"]: item for item in json.load(f)}
except FileNotFoundError:
  sys.exit(f"Dataset not found: {args.dataset}")

try:
  with open(args.results) as f:
    results = json.load(f)
except FileNotFoundError:
  sys.exit(f"Results file not found: {args.results}")
except json.JSONDecodeError as e:
  sys.exit(f"Failed to parse {args.results}: {e}")

if not isinstance(results, list):
  sys.exit("Results JSON must be a list of objects.")

print(f"Scoring {len(results)} entries against {len(gt_map)} papers in public dataset.\n")

# ──────────────────────────────────────────────────────────────────────────────
# Score
# ──────────────────────────────────────────────────────────────────────────────
per_paper: list[dict] = []
evid_total = corr_total = 0.0

for idx, item in enumerate(results, 1):
  title    = item.get("title", "")
  answer   = item.get("answer", "")
  evidence = item.get("evidence", [])

  gt = gt_map.get(title)
  if gt is None:
    print(f"[{idx:>3}] ⚠ title not found in public dataset — skipping")
    continue

  evid_score = calc_evidence_score(evidence, gt["evidence"])
  corr_score = judge_correctness(title, gt["question"], gt["evidence"], gt["answer"], answer, args.times)

  evid_total += evid_score
  corr_total += corr_score
  n = len(per_paper) + 1

  print(
    f"[{idx:>3}] evid={evid_score:.4f}  corr={corr_score:.0f}"
    f"   (running avg  evid={evid_total/n:.4f}  corr={corr_total/n:.4f})"
  )

  per_paper.append({
    "title":          title,
    "evidence_score": evid_score,
    "correctness":    corr_score,
    "answer":         answer,
  })

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
n = len(per_paper)
if n == 0:
  sys.exit("No papers scored.")

avg_evid = evid_total / n
avg_corr = corr_total / n

print(f"\n{'─'*60}")
print(f"  Papers scored  : {n} / {len(results)}")
print(f"  Evidence Score : {avg_evid:.5f}   (ROUGE-L, same as TA)")
print(f"  Correctness    : {avg_corr:.5f}   (3B judge, approximate)")
print(f"{'─'*60}")
print(
  "  ⚠  Correctness is an approximation.\n"
  "     TA grading uses a separate model with a different evaluation procedure.\n"
  "     Use this score as a development signal, not as a prediction of your final grade."
)

# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────
out_path = str(Path(args.results).with_suffix("")) + "_score.json"
with open(out_path, "w", encoding="utf-8") as f:
  json.dump({
    "summary": {
      "n_scored":       n,
      "n_total":        len(results),
      "evidence_score": round(avg_evid, 6),
      "correctness":    round(avg_corr, 6),
    },
    "per_paper": per_paper,
  }, f, indent=2, ensure_ascii=False)
print(f"\nSaved → {out_path}")
