"""Prompt templates for hallucination type classification in academic paper reviews.

Compact version: system prompt ~700 chars (~200 tokens), user structure ~300 chars (~85 tokens).
With evidence 1000 chars (~285 tokens) + answer ~5 tokens → total ~575 tokens → fits max_length=768.
"""

from functools import lru_cache
from pathlib import Path

SYSTEM_PROMPT = """Classify the hallucination type in a peer review sentence given paper evidence.

Hallucination types:
0. Attribution Failure — correct content but wrong source/section/citation
1. Entity — swapped or wrong named entity, method, dataset, or noun
2. Number — any numeric value differs (counts, %, years, dimensions, etc.)
3. Overgeneralization — accurate but overly broad; uses "all/every/always" or extends beyond paper scope
4. Temporal — wrong tense, modality (might vs. will), or time reference

Reply with EXACTLY ONE of these strings, nothing else:
Attribution Failure
Entity
Number
Overgeneralization
Temporal"""


def build_user_prompt(review_sentence: str, evidence: str) -> str:
    return (
        f"Evidence:\n{evidence}\n\n"
        f"Review Sentence:\n{review_sentence}\n\n"
        f"Hallucination type:"
    )


_LABEL2ID = {
    "attribution failure": 0,
    "entity": 1,
    "number": 2,
    "overgeneralization": 3,
    "temporal": 4,
}


def parse_model_output(output_text: str) -> int:
    text = output_text.strip().lower()
    if text in _LABEL2ID:
        return _LABEL2ID[text]
    best_pos = -1
    best_id = 0
    for name, idx in _LABEL2ID.items():
        pos = text.find(name)
        if pos == -1:
            continue
        if best_pos == -1 or pos < best_pos:
            best_pos = pos
            best_id = idx
    return best_id


_ID2LABEL = {v: k.title() for k, v in _LABEL2ID.items()}

_FEW_SHOT_IDS = [1129, 3401, 2007, 5588, 4114]
_DEFAULT_CSV_PATH = Path(__file__).parent / "train_processed.csv"
_FEW_SHOT_EVIDENCE_CHAR_CAP = 600  # 原本 1200，精簡以節省 token


@lru_cache(maxsize=1)
def _load_few_shot_examples() -> tuple[tuple[int, str, str], ...]:
    import pandas as pd
    df = pd.read_csv(_DEFAULT_CSV_PATH)
    rows = []
    for fs_id in _FEW_SHOT_IDS:
        row = df.loc[df["id"] == fs_id].iloc[0]
        evidence = str(row["evidence"])
        if len(evidence) > _FEW_SHOT_EVIDENCE_CHAR_CAP:
            evidence = evidence[:_FEW_SHOT_EVIDENCE_CHAR_CAP].rstrip() + " ..."
        rows.append((int(row["label"]), str(row["text"]), evidence))
    return tuple(rows)


def format_chat_messages(
    review_sentence: str,
    evidence: str,
    few_shot: bool = False,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        for label_id, ex_text, ex_evidence in _load_few_shot_examples():
            messages.append(
                {"role": "user", "content": build_user_prompt(ex_text, ex_evidence)}
            )
            messages.append(
                {"role": "assistant", "content": _ID2LABEL[label_id]}
            )
    messages.append(
        {"role": "user", "content": build_user_prompt(review_sentence, evidence)}
    )
    return messages