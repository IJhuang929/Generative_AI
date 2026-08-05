"""Run LoRA-adapter inference and write the final submission CSV.

End-to-end pipeline (per HW3 spec, ``python inference.py`` must produce the
final result without any pre-generated static files):

    {split}.csv + paper_evidence/{split}/*.pdf
        -> PDF parse + RAG retrieval (inlined from former preprocess.py)
        -> chat-template prompt (inlined from former prompt_template.py)
        -> LoRA-adapter generate (class name)
        -> parse class id (0..4)
        -> hw3_{student_id}.csv (id,label)

prompt_template.py / pdf_parser.py / preprocess.py have been merged into
this file so the submission folder contains exactly the files listed in
the HW3 spec.
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Optional

# Unsloth must be imported before transformers / torch so its monkey-patches
# land before the LM stack is touched.
from unsloth import FastLanguageModel  # type: ignore[import-not-found]

import fitz
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# =========================================================================
# Sentence-transformer embedder (formerly pdf_parser.py)
# =========================================================================

_EMBEDDER = None
_EMBEDDER_NAME = os.environ.get(
    "HW3_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2"
)


def _get_embedder():
    """Lazy-load the sentence-transformers model (singleton)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[pdf_parser] loading embedder {_EMBEDDER_NAME} on {device}...")
        _EMBEDDER = SentenceTransformer(_EMBEDDER_NAME, device=device)
    return _EMBEDDER


def release_embedder() -> None:
    """Free the cached embedder and reclaim its GPU memory before the LM loads."""
    global _EMBEDDER
    if _EMBEDDER is None:
        return
    try:
        _EMBEDDER.to("cpu")
    except Exception:
        pass
    del _EMBEDDER
    _EMBEDDER = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode strings into an L2-normalized embedding matrix (cosine = dot)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = _get_embedder()
    return np.asarray(
        model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )


# =========================================================================
# PDF text extraction + paragraph segmentation (formerly pdf_parser.py)
# =========================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract plain text from a PDF, dropping page-number / header / footer noise."""
    if not os.path.isfile(pdf_path):
        return ""
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return ""

    cleaned_pages: list[str] = []
    try:
        for page in doc:
            page_text = page.get_text("text")
            # Strip C0 control chars (except \t \n \r) so corrupt fonts don't
            # break csv writers downstream.
            page_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", page_text)
            kept_lines: list[str] = []
            for raw_line in page_text.splitlines():
                line = raw_line.strip()
                if not line:
                    kept_lines.append("")
                    continue
                if re.fullmatch(r"[-\s\d]+", line):
                    continue
                if len(line) < 15:
                    continue
                kept_lines.append(line)
            cleaned_pages.append("\n".join(kept_lines))
    finally:
        doc.close()
    return "\n\n".join(cleaned_pages)


def split_into_paragraphs(text: str) -> list[str]:
    """Split text on blank lines and discard paragraphs shorter than 50 chars."""
    if not text:
        return []
    chunks = re.split(r"\n\s*\n+", text)
    paragraphs: list[str] = []
    for chunk in chunks:
        normalized = re.sub(r"\s*\n\s*", " ", chunk).strip()
        if len(normalized) >= 50:
            paragraphs.append(normalized)
    return paragraphs


def retrieve_relevant_paragraphs(
    review_sentence: str,
    paragraphs: list[str],
    top_k: int = 5,
    paragraph_embeddings: Optional[np.ndarray] = None,
) -> str:
    """Return the top-k cosine-most-similar paragraphs joined by blank lines."""
    if not paragraphs or not review_sentence:
        return ""
    k = max(0, min(top_k, len(paragraphs)))
    if k == 0:
        return ""

    if paragraph_embeddings is None:
        paragraph_embeddings = encode_texts(paragraphs)
    query_emb = encode_texts([review_sentence])

    sims = cosine_similarity(query_emb, paragraph_embeddings).ravel()
    top_indices = sims.argsort()[::-1][:k]
    selected = [paragraphs[i] for i in top_indices]
    return "\n\n".join(selected)


# =========================================================================
# Per-row evidence builder + split loader (formerly preprocess.py)
# =========================================================================

def build_evidence_column(
    df: pd.DataFrame,
    pdf_dir: str,
    top_k: int = 2,
    max_chars: int = 1000,
    desc: str | None = None,
) -> pd.DataFrame:
    """Append an `evidence` column to df via per-paper PDF parse + retrieval.

    PDF parse and paragraph encoding are cached per `paper_id` so each PDF is
    processed only once even though many rows share the same paper.
    """
    paragraphs_cache: dict[str, list[str]] = {}
    embedding_cache: dict[str, np.ndarray] = {}
    evidences: list[str] = []

    iterator = tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc=desc or "build_evidence",
    )
    for row in iterator:
        paper_id = str(row.paper_id)
        claim = str(row.text)

        try:
            if paper_id not in paragraphs_cache:
                pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
                if not os.path.isfile(pdf_path):
                    print(
                        f"[warn] PDF not found: {pdf_path} (id={row.id})",
                        file=sys.stderr,
                    )
                    paragraphs_cache[paper_id] = []
                    embedding_cache[paper_id] = np.zeros((0, 0), dtype=np.float32)
                else:
                    raw_text = extract_text_from_pdf(pdf_path)
                    paragraphs = split_into_paragraphs(raw_text)
                    paragraphs_cache[paper_id] = paragraphs
                    embedding_cache[paper_id] = encode_texts(paragraphs)

            paragraphs = paragraphs_cache[paper_id]
            if not paragraphs:
                evidences.append("")
                continue

            evidence = retrieve_relevant_paragraphs(
                claim,
                paragraphs,
                top_k=top_k,
                paragraph_embeddings=embedding_cache[paper_id],
            )
            if len(evidence) > max_chars:
                evidence = evidence[:max_chars]
            evidences.append(evidence)

        except Exception as exc:
            print(
                f"[warn] failed to build evidence for id={row.id} "
                f"paper_id={paper_id}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            evidences.append("")

    out = df.copy()
    out["evidence"] = evidences
    return out


def load_split_with_evidence(
    data_dir: str,
    split: str,
    top_k: int = 2,
    max_chars: int = 1000,
) -> pd.DataFrame:
    """Read {split}.csv and parse PDFs on the fly to attach an `evidence` column.

    Per HW3 spec, no pre-generated static CSV is read or written — every run
    rebuilds evidence directly from `paper_evidence/{split}/*.pdf`.
    """
    raw_path = os.path.join(data_dir, f"{split}.csv")
    pdf_dir = os.path.join(data_dir, "paper_evidence", split)

    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"Input CSV not found: {raw_path}")
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    df_raw = pd.read_csv(raw_path)
    print(f"[evidence] building from {raw_path} + {pdf_dir}/...")
    return build_evidence_column(
        df_raw, pdf_dir, top_k=top_k, max_chars=max_chars, desc=f"{split}.csv",
    )


# =========================================================================
# Chat prompt template (formerly prompt_template.py)
# =========================================================================

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
    """Map a free-form model completion to a class id in [0..4]."""
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


def format_chat_messages(review_sentence: str, evidence: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(review_sentence, evidence)},
    ]


# =========================================================================
# CLI + inference loop
# =========================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LoRA-adapter inference and emit the submission CSV.",
    )
    parser.add_argument("--data_dir", type=str, default=".",
                        help="Directory containing {split}.csv and "
                             "paper_evidence/{split}/*.pdf.")
    parser.add_argument("--adapter_dir", type=str, default="./adapter_checkpoint",
                        help="Path to the trained LoRA adapter directory.")
    parser.add_argument("--output_file", type=str, default="./hw3_314605011.csv",
                        help="Where to write the predictions CSV.")
    parser.add_argument("--max_length", type=int, default=600,
                        help="Maximum tokenized prompt length.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Inference batch size.")
    parser.add_argument("--split", type=str, default="test", choices=["test", "dev"],
                        help="Which split to run inference on.")
    parser.add_argument("--evidence_top_k", type=int, default=2,
                        help="Top-k paragraphs to retrieve per row.")
    parser.add_argument("--evidence_max_chars", type=int, default=1000,
                        help="Per-row evidence character cap.")
    return parser


def batch_inference(
    df: pd.DataFrame,
    batch_size: int,
    *,
    model,
    tokenizer,
    max_length: int,
    max_new_tokens: int = 8,
) -> list[int]:
    """Padded-batch greedy generation; parses each completion into a class id.

    Causal-LM batched generation requires *left*-padding so every prompt's
    last token sits at the same position before generation begins. We
    temporarily flip ``tokenizer.padding_side = "left"`` for the duration
    of this call and restore it afterwards.
    """
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    preds: list[int] = []
    try:
        n = len(df)
        pbar = tqdm(range(0, n, batch_size), desc="inference")
        for start in pbar:
            end = min(start + batch_size, n)
            batch_rows = df.iloc[start:end]
            prompts = [
                tokenizer.apply_chat_template(
                    format_chat_messages(str(r["text"]), str(r["evidence"])),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for _, r in batch_rows.iterrows()
            ]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                )

            input_len = enc["input_ids"].shape[1]
            new_tokens = out[:, input_len:]
            decoded_batch = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for raw in decoded_batch:
                preds.append(parse_model_output(raw))

            if end // 100 != start // 100:
                dist = Counter(preds)
                dist_str = ", ".join(f"{c}:{dist.get(c, 0)}" for c in range(5))
                pbar.write(f"[{end}/{n}] pred dist -> {dist_str}")
    finally:
        tokenizer.padding_side = original_padding_side

    return preds


def main() -> None:
    args = _build_arg_parser().parse_args()

    print(f"[1/3] Building evidence for split={args.split} from {args.data_dir}/...")
    df = load_split_with_evidence(
        args.data_dir,
        args.split,
        top_k=args.evidence_top_k,
        max_chars=args.evidence_max_chars,
    )
    print(f"  {len(df)} rows")
    # MiniLM is no longer needed; free its ~80MB before the LM is loaded.
    release_embedder()

    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

    print(f"[2/3] Loading adapter from {args.adapter_dir} (4-bit, max_seq_length={args.max_length})...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_dir,
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)

    print(f"[3/3] Running inference on {len(df)} rows...")
    preds = batch_inference(
        df,
        args.batch_size,
        model=model,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    out_df = pd.DataFrame({"id": df["id"].values, "label": preds})
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    if args.split == "dev" and "label" in df.columns:
        from sklearn.metrics import classification_report, f1_score

        true_labels = df["label"].astype(int).tolist()
        label_ids = list(range(5))
        class_names = [
            "Attribution Failure", "Entity", "Number",
            "Overgeneralization", "Temporal",
        ]

        macro_f1 = float(f1_score(
            true_labels, preds,
            average="macro", labels=label_ids, zero_division=0,
        ))

        print("=" * 40)
        print(f"Dev Macro F1: {macro_f1:.4f}")
        print("=" * 40)
        print("Per-class report:")
        print(classification_report(
            true_labels, preds,
            labels=label_ids, target_names=class_names,
            digits=4, zero_division=0,
        ))

        pred_dist = Counter(preds)
        true_dist = Counter(true_labels)
        print("Distribution (pred vs true):")
        print(f"{'class':<22} {'pred':>6} {'true':>6}")
        for c, name in enumerate(class_names):
            print(f"{c} {name:<20} {pred_dist.get(c, 0):>6} {true_dist.get(c, 0):>6}")

    print("=" * 40)
    print("Inference Complete!")
    print(f"Split        : {args.split}")
    print(f"Total samples: {len(out_df)}")
    print(f"Output saved : {out_path}")
    print("=" * 40)


if __name__ == "__main__":
    main()
