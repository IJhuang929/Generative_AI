"""PDF parsing and evidence retrieval utilities for the paper-claim classification task.

Uses PyMuPDF (fitz) for text extraction and a sentence-transformers model
(``sentence-transformers/all-MiniLM-L6-v2`` by default — open-source, ~22M
params, 384-d embeddings) for semantic retrieval. Cosine similarity ranks
paragraphs against the review sentence (claim).
"""

from __future__ import annotations

import os
import re
from typing import Optional

import fitz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Lazy-loaded module-level singleton so loading the encoder happens at most
# once per process (about 80 MB on disk + a few hundred MB of RAM).
_EMBEDDER = None
_EMBEDDER_NAME = os.environ.get(
    "HW3_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2"
)


def _get_embedder():
    """Return the sentence-transformers model, loading it on first call."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[pdf_parser] loading embedder {_EMBEDDER_NAME} on {device}...")
        _EMBEDDER = SentenceTransformer(_EMBEDDER_NAME, device=device)
    return _EMBEDDER


def release_embedder() -> None:
    """Free the cached sentence-transformers model and reclaim its GPU memory.

    Safe to call even if the embedder was never loaded. Useful after
    preprocessing finishes and we don't want the ~80 MB embedder lingering
    on the GPU while the larger LM is being trained / used for inference.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        return
    try:
        _EMBEDDER.to("cpu")
    except Exception:
        pass
    del _EMBEDDER
    _EMBEDDER = None

    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def encode_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode a list of strings into a (N, D) L2-normalized embedding matrix.

    The returned vectors are L2-normalized so that the dot product between
    any two rows equals their cosine similarity.

    Args:
        texts: List of strings to encode.
        batch_size: Encoder mini-batch size.

    Returns:
        ``np.ndarray`` of shape ``(len(texts), embedding_dim)``, dtype float32.
        Returns an empty ``(0, 0)`` array if ``texts`` is empty.
    """
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


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract the full plain text of a PDF, dropping short header/footer/page-number lines.

    Args:
        pdf_path: Filesystem path to the PDF file.

    Returns:
        The concatenated plain text of every page, with very short lines
        (typically page numbers or running headers/footers) removed.
        Returns an empty string if the file cannot be opened.
    """
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
            # Strip NUL and other C0 control chars (except \t \n \r). These
            # can leak in from corrupt fonts / annotations and break csv writers.
            page_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", page_text)
            kept_lines: list[str] = []
            for raw_line in page_text.splitlines():
                line = raw_line.strip()
                if not line:
                    kept_lines.append("")
                    continue
                # Drop pure page numbers like "12" or "- 3 -".
                if re.fullmatch(r"[-\s\d]+", line):
                    continue
                # Drop very short lines (likely headers/footers/page markers).
                if len(line) < 15:
                    continue
                kept_lines.append(line)
            cleaned_pages.append("\n".join(kept_lines))
    finally:
        doc.close()

    return "\n\n".join(cleaned_pages)


def split_into_paragraphs(text: str) -> list[str]:
    """Split plain text into paragraphs and discard short ones.

    Paragraph boundaries are defined as one or more blank lines. Paragraphs
    shorter than 50 characters (after stripping) are filtered out.

    Args:
        text: Raw plain text, typically the output of `extract_text_from_pdf`.

    Returns:
        A list of paragraph strings, each at least 50 characters long.
    """
    if not text:
        return []

    chunks = re.split(r"\n\s*\n+", text)
    paragraphs: list[str] = []
    for chunk in chunks:
        # Collapse single newlines inside a paragraph into spaces so that
        # PDF line-wrapping does not fragment the text.
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
    """Retrieve the top-k paragraphs most similar to a review sentence via dense embeddings.

    Embeds the query and (optionally pre-encoded) paragraphs with the
    module-level sentence-transformers model, then returns the top-k
    paragraphs by cosine similarity, joined by blank lines.

    Args:
        review_sentence: The claim sentence to match evidence against.
        paragraphs: Candidate paragraphs from the source paper.
        top_k: Maximum number of paragraphs to return.
        paragraph_embeddings: Optional pre-computed (N, D) matrix of L2-
            normalized paragraph embeddings (e.g. cached per paper). When
            None, paragraphs are encoded on the fly.

    Returns:
        The selected paragraphs joined by "\\n\\n", in descending similarity order.
        Empty string if ``paragraphs`` is empty.
    """
    if not paragraphs or not review_sentence:
        return ""

    # Clamp top_k to the available paragraph count (and guard against <= 0)
    # so downstream slicing/argsort never goes out of bounds.
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


def get_evidence(
    review_sentence: str,
    pdf_path: str,
    top_k: int = 2,
    max_chars: int = 1000,
) -> str:
    """End-to-end evidence retrieval: PDF -> paragraphs -> top-k -> truncated string.

    Args:
        review_sentence: The claim sentence whose evidence we want.
        pdf_path: Path to the source PDF.
        top_k: Number of top paragraphs to keep.
        max_chars: Hard cap on the length of the returned string.

    Returns:
        Up to `max_chars` characters of the most relevant paragraphs concatenated,
        or an empty string if the PDF is missing or yields no usable text.
    """
    if not os.path.isfile(pdf_path):
        return ""

    text = extract_text_from_pdf(pdf_path)
    if not text:
        return ""

    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        return ""

    evidence = retrieve_relevant_paragraphs(review_sentence, paragraphs, top_k=top_k)
    if len(evidence) > max_chars:
        evidence = evidence[:max_chars]
    return evidence
