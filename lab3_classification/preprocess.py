"""Pre-compute per-row PDF evidence for the paper-claim classification task.

For each row in train.csv / dev.csv / test.csv, locates the corresponding paper
PDF under paper_evidence/{split}/{paper_id}.pdf, retrieves the top-k most
relevant paragraphs for the row's claim, and writes the augmented CSV with a
new `evidence` column.

PDF text + paragraph segmentation is cached per `paper_id` so each PDF is
parsed only once per split, even though many rows share the same paper.

The reusable :func:`build_evidence_column` is imported by ``train.py`` and
``inference.py`` so PDF parsing + retrieval runs end-to-end inside those
entry points (per the HW3 spec: ``python inference.py`` must produce the
final result without relying on pre-generated static CSVs).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import pandas as pd
from tqdm import tqdm

import numpy as np

from pdf_parser import (
    encode_texts,
    extract_text_from_pdf,
    retrieve_relevant_paragraphs,
    split_into_paragraphs,
)


SPLITS = ("train", "dev", "test")


def build_evidence_column(
    df: pd.DataFrame,
    pdf_dir: str,
    top_k: int = 2,
    max_chars: int = 1000,
    desc: str | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with an ``evidence`` column appended.

    For each row, looks up ``{pdf_dir}/{paper_id}.pdf``, parses + paragraph-
    segments it (cached per paper_id), then retrieves the top-k paragraphs
    most relevant to ``row.text`` and truncates to ``max_chars``.

    Args:
        df: DataFrame containing at least ``id``, ``paper_id``, ``text``.
        pdf_dir: Directory containing ``{paper_id}.pdf`` files for this split.
        top_k: Number of top paragraphs per row.
        max_chars: Hard cap on the per-row evidence string length.
        desc: Optional progress-bar description.

    Returns:
        A new DataFrame with the same rows + an ``evidence`` column. Rows
        whose PDF is missing or unreadable get an empty ``evidence`` string.
    """
    # paper_id -> (paragraphs, embeddings). Both PDF parse and paragraph
    # encoding are expensive — many rows share the same paper_id, so cache
    # both per paper and reuse across rows.
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
    cache: bool = True,
) -> pd.DataFrame:
    """Load ``{split}.csv`` and attach the ``evidence`` column on the fly.

    If ``cache=True`` and ``{split}_processed.csv`` already exists with a
    matching row count, it is reused. Otherwise PDFs are parsed on the spot
    and (optionally) the result is written to ``{split}_processed.csv`` for
    future runs.

    Args:
        data_dir: Directory containing ``{split}.csv`` and ``paper_evidence/``.
        split: One of ``"train"``, ``"dev"``, ``"test"``.
        top_k: Top paragraphs per row.
        max_chars: Per-row evidence character cap.
        cache: If True, prefer/save the processed CSV cache file.

    Returns:
        DataFrame with original columns + ``evidence``.
    """
    raw_path = os.path.join(data_dir, f"{split}.csv")
    cache_path = os.path.join(data_dir, f"{split}_processed.csv")
    pdf_dir = os.path.join(data_dir, "paper_evidence", split)

    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"Input CSV not found: {raw_path}")

    df_raw = pd.read_csv(raw_path)

    if cache and os.path.isfile(cache_path):
        df_cached = pd.read_csv(cache_path)
        if len(df_cached) == len(df_raw) and "evidence" in df_cached.columns:
            print(f"[evidence] reusing cache {cache_path} ({len(df_cached)} rows)")
            return df_cached
        print(
            f"[evidence] cache {cache_path} is stale "
            f"(rows {len(df_cached)} vs {len(df_raw)}); rebuilding."
        )

    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    print(f"[evidence] building from {raw_path} + {pdf_dir}/...")
    df_with_ev = build_evidence_column(
        df_raw, pdf_dir, top_k=top_k, max_chars=max_chars, desc=f"{split}.csv",
    )

    if cache:
        try:
            df_with_ev.to_csv(cache_path, index=False, escapechar="\\")
            print(f"[evidence] cached -> {cache_path}")
        except (OSError, PermissionError) as exc:
            # Read-only data_dir (e.g. Kaggle's /kaggle/input mount) — skip
            # caching rather than crashing. Subsequent runs will rebuild,
            # which is the correct behavior for a read-only source.
            print(
                f"[evidence] cache write skipped ({exc.__class__.__name__}: "
                f"{exc}); continuing without cache.",
                file=sys.stderr,
            )
    return df_with_ev


def process_split(
    csv_path: str,
    pdf_dir: str,
    out_path: str,
    top_k: int = 2,
    max_chars: int = 1000,
) -> None:
    """Process one CSV split: append an `evidence` column and write to disk."""
    df = pd.read_csv(csv_path)
    df = build_evidence_column(
        df, pdf_dir, top_k=top_k, max_chars=max_chars,
        desc=os.path.basename(csv_path),
    )
    # escapechar is a belt-and-suspenders safety net: even if exotic control
    # chars survive sanitisation, the csv writer won't blow up.
    df.to_csv(out_path, index=False, escapechar="\\")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute per-row PDF evidence for train/dev/test."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory containing the CSV files and the paper_evidence/ folder.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of top paragraphs to retrieve per row.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1000,
        help="Maximum number of characters in the evidence column.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        choices=SPLITS,
        help="Which splits to process (default: all).",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)

    for split in args.splits:
        csv_path = os.path.join(data_dir, f"{split}.csv")
        pdf_dir = os.path.join(data_dir, "paper_evidence", split)
        out_path = os.path.join(data_dir, f"{split}_processed.csv")

        if not os.path.isfile(csv_path):
            print(f"[warn] missing {csv_path}, skipping {split}.", file=sys.stderr)
            continue
        if not os.path.isdir(pdf_dir):
            print(f"[warn] missing {pdf_dir}, skipping {split}.", file=sys.stderr)
            continue

        print(f"==> Processing {split}: {csv_path}")
        process_split(
            csv_path=csv_path,
            pdf_dir=pdf_dir,
            out_path=out_path,
            top_k=args.top_k,
            max_chars=args.max_chars,
        )
        print(f"    wrote {out_path}")


if __name__ == "__main__":
    main()
