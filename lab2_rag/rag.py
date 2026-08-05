"""
HW2: Document QA based on RAG
生成式人工智慧 - 2026 Spring

LLM  : meta-llama/Llama-3.2-3B-Instruct
Embed: BAAI/bge-large-en-v1.5
Rerank: BAAI/bge-reranker-v2-m3

Single-file submission consolidating:
    config.py, models.py, text_processor.py, main.py

vLLM server (run before executing this script):
    # 24GB VRAM (float16, tested on RTX 4090):
    uv run vllm serve meta-llama/Llama-3.2-3B-Instruct \
        --port 8091 \
        --api-key abc \
        --dtype float16 \
        --gpu-memory-utilization 0.4 \
        --max-model-len 25000

    # 8GB VRAM (bitsandbytes 4-bit, tested on RTX 3070 Ti)
    # [code: 53] modify USE_FP16_MODELS = True:
    uv run vllm serve meta-llama/Llama-3.2-3B-Instruct \
        --port 8091 --api-key abc --dtype float16 \
        --quantization bitsandbytes --load-format bitsandbytes \
        --gpu-memory-utilization 0.42 --max-model-len 3072
"""

import gc
import json
import os
import re
import subprocess

import numpy as np
import openai
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder


# ── Configuration ─────────────────────────────────────────────────────────────

HF_TOKEN          = "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"
LLM_MODEL_ID      = "meta-llama/Llama-3.2-3B-Instruct"
VLLM_BASE_URL     = "http://localhost:8091/v1"
VLLM_API_KEY      = "abc"
EMBED_MODEL_ID    = "BAAI/bge-large-en-v1.5"
RERANK_MODEL_ID   = "BAAI/bge-reranker-v2-m3"
USE_PRIVATE       = True   # True for private dataset, False for public dataset
USE_HYDE          = False
USE_FP16_MODELS   = False   # FP16 embed + rerank (~1.7 GB). Set False for FP32 (~3.5 GB)
PUBLIC_DATA_PATH  = "public_dataset.json"
PRIVATE_DATA_PATH = "private_dataset.json"
PRIVATE_DATA_ANS_PATH = "private_dataset_answer.json"
OUTPUT_PATH       = "result.json"

EMBED_CACHE_DIR    = "embed_cache"

CHILD_CHUNK_SIZE    = 320
CHILD_CHUNK_OVERLAP = 64
PARENT_CHUNK_SIZE   = 1024
PARENT_CHUNK_OVERLAP = 256

TOP_K            = 15
RERANK_THRESHOLD = 0.25
EVIDENCE_TOP_K   = 2
LLM_TOP_K        = 4

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def default_cfg() -> dict:
    return {
        "CHILD_CHUNK_SIZE":    CHILD_CHUNK_SIZE,
        "CHILD_CHUNK_OVERLAP": CHILD_CHUNK_OVERLAP,
        "PARENT_CHUNK_SIZE":   PARENT_CHUNK_SIZE,
        "PARENT_CHUNK_OVERLAP": PARENT_CHUNK_OVERLAP,
        "TOP_K":               TOP_K,
        "LLM_TOP_K":           LLM_TOP_K,
        "RERANK_THRESHOLD":    RERANK_THRESHOLD,
        "EVIDENCE_TOP_K":      EVIDENCE_TOP_K,
        "USE_HYDE":            USE_HYDE,
    }


# ── Model Initialization ──────────────────────────────────────────────────────

_llm_client = openai.OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

_embed_kwargs = {"device": "cuda"}
_rerank_kwargs = {"device": "cuda"}
if USE_FP16_MODELS:
    _embed_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    _rerank_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_ID,
    model_kwargs=_embed_kwargs,
    encode_kwargs={"normalize_embeddings": True},
)
print("✓ Embedding model loaded")

print("Loading reranker model...")
reranker = CrossEncoder(RERANK_MODEL_ID, **_rerank_kwargs)
print("✓ Reranker loaded")


def get_llm_client():
    return _llm_client

def get_embedding_model():
    return embedding_model

def get_reranker():
    return reranker


# ── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'BIBREF\d+', '', text)
    text = re.sub(r'INLINEFORM\d+', '', text)
    text = re.sub(r'TABREF\d+', '', text)
    text = re.sub(r'FIGREF\d+', '', text)
    text = re.sub(r'SECREF\d+', '', text)
    text = re.sub(r'DISPLAYFORM\d+', '', text)
    text = re.sub(r'EQREF\d+', '', text)
    text = re.sub(r'FORMREF\d+', '', text)
    text = re.sub(r'UID\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Splitters ─────────────────────────────────────────────────────────────────

def make_splitters(cfg: dict):
    child = RecursiveCharacterTextSplitter(
        chunk_size=cfg["CHILD_CHUNK_SIZE"],
        chunk_overlap=cfg["CHILD_CHUNK_OVERLAP"],
        separators=["\n\n", "\n", " ", ""],
    )
    parent = RecursiveCharacterTextSplitter(
        chunk_size=cfg["PARENT_CHUNK_SIZE"],
        chunk_overlap=cfg["PARENT_CHUNK_OVERLAP"],
        separators=["\n\n", "\n", " ", ""],
    )
    return child, parent


# ── Prompt Helpers ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You extract answers from scientific papers. "
    "Rules: "
    "1. Copy exact phrases from the context verbatim — do NOT paraphrase. "
    "2. Output ONLY the answer. No preamble like 'Based on', 'The answer is', 'According to'. "
    "3. For entities (name/number/method): output only the entity, no sentence. "
    "4. For lists: output comma-separated items as they appear in the context. "
    "5. For explanations: 1 sentence max, quoting context phrases. "
    "6. NEVER say you don't know or that context is insufficient — always extract the best possible answer from the context."
)

FEW_SHOT_EXAMPLES = [
    {
        "context": "Therefore, we used the crowdsourcing platform CrowdFlower (CF) for our data collection.",
        "question": "How was this data collected?",
        "answer": "CrowdFlower",
    },
    {
        "context": "The SemEval-2010 benchmark dataset BIBREF0 is composed of 244 scientific articles collected from the ACM Digital Library (conference and workshop papers).",
        "question": "how many articles are in the dataset?",
        "answer": "244 ",
    },
    {
        "context": "We only used models of dimension 300 and, for Word2Vec, Wang2Vec, and FastText, skip-gram architectured models.",
        "question": "what word embedding techniques did they experiment with?",
        "answer": "Word2Vec, Wang2Vec, and FastText",
    },
]

PREFIXES = [
    "the answer is", "based on the context", "based on the provided context",
    "according to the context", "according to the passage", "in the context,",
    "the context states", "from the context,", "answer:", "a:",
]


def _strip_preface(s: str) -> str:
    low = s.lower().lstrip()
    for p in PREFIXES:
        if low.startswith(p):
            s = s[len(p):].lstrip(" ,:.-")
            low = s.lower().lstrip()
    return s.strip().strip('"').strip("'")


def build_user_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer directly and concisely:"
    )


def build_prompt(question: str, context_chunks: list[str]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user",      "content": f"Context:\n{ex['context']}\n\nQuestion: {ex['question']}\n\nAnswer directly and concisely:"})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": build_user_prompt(question, context_chunks)})
    return messages


# ── Extractive Prompt ─────────────────────────────────────────────────────────

EXTRACTIVE_SYSTEM_PROMPT = (
    "You are an extractive QA system. "
    "Find which numbered passage best answers the question, then copy verbatim. "
    "Output ONLY:\n"
    "PASSAGE: <number>\n"
    "EVIDENCE: <exact sentence from the passage that supports the answer>\n"
    "ANSWER: <exact verbatim phrase from the passage>"
)

_EXTRACTIVE_FEW_SHOTS = [
    {
        "context": "[1] Therefore, we used the crowdsourcing platform CrowdFlower (CF) for our data collection.",
        "question": "How was this data collected?",
        "answer": "PASSAGE: 1\nEVIDENCE: Therefore, we used the crowdsourcing platform CrowdFlower (CF) for our data collection.\nANSWER: CrowdFlower",
    },
    {
        "context": "[1] The SemEval-2010 benchmark dataset is composed of 244 scientific articles collected from the ACM Digital Library.",
        "question": "how many articles are in the dataset?",
        "answer": "PASSAGE: 1\nEVIDENCE: The SemEval-2010 benchmark dataset is composed of 244 scientific articles collected from the ACM Digital Library.\nANSWER: 244",
    },
    {
        "context": "[1] We only used models of dimension 300 and, for Word2Vec, Wang2Vec, and FastText, skip-gram architectured models.",
        "question": "what word embedding techniques did they experiment with?",
        "answer": "PASSAGE: 1\nEVIDENCE: We only used models of dimension 300 and, for Word2Vec, Wang2Vec, and FastText, skip-gram architectured models.\nANSWER: Word2Vec, Wang2Vec, and FastText",
    },
]


def build_extractive_prompt(question: str, chunks: list[str]) -> list[dict]:
    numbered = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
    messages = [{"role": "system", "content": EXTRACTIVE_SYSTEM_PROMPT}]
    for ex in _EXTRACTIVE_FEW_SHOTS:
        messages.append({"role": "user",      "content": f"{ex['context']}\n\nQuestion: {ex['question']}"})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": f"{numbered}\n\nQuestion: {question}"})
    return messages


# ── Cache Helpers ─────────────────────────────────────────────────────────────

def chunk_cfg_key(cfg: dict) -> str:
    return (
        f"cs{cfg['CHILD_CHUNK_SIZE']}_co{cfg['CHILD_CHUNK_OVERLAP']}"
        f"_ps{cfg['PARENT_CHUNK_SIZE']}_po{cfg['PARENT_CHUNK_OVERLAP']}"
    )


def dataset_tag(data_path: str) -> str:
    return "private" if data_path == PRIVATE_DATA_PATH else "public"


def prebuild_embedding_cache(dataset: list[dict], cfg: dict, tag: str):
    embedding_model = get_embedding_model()
    chunk_key = chunk_cfg_key(cfg)
    base_dir = os.path.join(EMBED_CACHE_DIR, tag, chunk_key)
    os.makedirs(base_dir, exist_ok=True)

    child_splitter, parent_splitter = make_splitters(cfg)
    total = len(dataset)
    for idx, item in enumerate(dataset):
        paper_dir    = os.path.join(base_dir, f"paper_{idx}")
        parents_path = os.path.join(base_dir, f"paper_{idx}_parents.json")
        if os.path.isdir(paper_dir) and os.path.exists(parents_path):
            continue

        full_text     = clean_text(item["full_text"])
        title         = item.get("title", "")
        child_chunks  = child_splitter.split_text(full_text)
        parent_chunks = parent_splitter.split_text(full_text)

        with open(parents_path, "w", encoding="utf-8") as f:
            json.dump(parent_chunks, f, ensure_ascii=False)

        children_path = os.path.join(base_dir, f"paper_{idx}_children.json")
        with open(children_path, "w", encoding="utf-8") as f:
            json.dump(child_chunks, f, ensure_ascii=False)

        if not child_chunks:
            os.makedirs(paper_dir, exist_ok=True)
            print(f"[cache] {idx+1}/{total} (empty)")
            continue

        title_prefix = f"Paper: {title}. Passage: "
        chunks_with_title = [title_prefix + c for c in child_chunks]
        vectorstore = FAISS.from_texts(chunks_with_title, embedding_model)
        vectorstore.save_local(paper_dir)
        print(f"[cache] {idx+1}/{total} ...")

        del vectorstore
        gc.collect()
        torch.cuda.empty_cache()


# ── Cache Loading ─────────────────────────────────────────────────────────────

def _load_cache(base_dir: str, idx: int):
    """Load vectorstore + parent/child chunks from cache. Returns (vs, pc, cc) or None."""
    paper_dir = os.path.join(base_dir, f"paper_{idx}")
    parents_path = os.path.join(base_dir, f"paper_{idx}_parents.json")
    if not (os.path.isdir(paper_dir) and os.path.exists(parents_path) and os.listdir(paper_dir)):
        return None
    vs = FAISS.load_local(paper_dir, embeddings=get_embedding_model(
    ), allow_dangerous_deserialization=True)
    with open(parents_path, "r", encoding="utf-8") as f:
        pc = json.load(f)
    children_path = os.path.join(base_dir, f"paper_{idx}_children.json")
    cc = json.load(open(children_path, encoding="utf-8")
                   ) if os.path.exists(children_path) else None
    return vs, pc, cc


# ── Retrieval Helpers ─────────────────────────────────────────────────────────

def _find_parent_chunk(child: str, parent_chunks: list[str]) -> str:
    for parent in parent_chunks:
        if child in parent:
            return parent
    best, best_overlap = child, 0
    child_words = set(child.split())
    for parent in parent_chunks:
        overlap = len(child_words & set(parent.split()))
        if overlap > best_overlap:
            best, best_overlap = parent, overlap
    return best


def _best_sentence_from_parent(child: str, parent_chunks: list[str]) -> str:
    """Return the single most relevant complete sentence from the parent chunk."""
    parent = _find_parent_chunk(child, parent_chunks)
    sentences = re.split(r'(?<=[.!?])\s+', parent)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if not sentences:
        return child
    child_words = set(child.lower().split())
    best = max(sentences, key=lambda s: len(
        set(s.lower().split()) & child_words))
    return best


# ── LLM Inference ─────────────────────────────────────────────────────────────

# Primary: full format match (relaxed whitespace)
_EXTRACTIVE_RE = re.compile(
    r'PASSAGE:\s*(\d+)\s+'
    r'EVIDENCE:\s*(.+?)\s+'
    r'ANSWER:\s*(.+)',
    re.IGNORECASE | re.DOTALL,
)
# Fallback: just EVIDENCE + ANSWER (no PASSAGE required)
_EVIDENCE_ANSWER_RE = re.compile(
    r'EVIDENCE:\s*(.+?)\s+ANSWER:\s*(.+)',
    re.IGNORECASE | re.DOTALL,
)
# Last resort: just ANSWER
_ANSWER_ONLY_RE = re.compile(r'ANSWER:\s*(.+)', re.IGNORECASE)


def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip().lower()


def _find_verbatim(quote: str, passages: list[str]) -> str | None:
    q = _norm(quote)
    for passage in passages:
        p = _norm(passage)
        pos = p.find(q)
        if pos != -1:
            return passage[pos:pos + len(quote)]
    return None


def _parse_extractive(raw: str, passages: list[str]) -> tuple[str, str | None] | None:
    # Stage 1: full format (PASSAGE + EVIDENCE + ANSWER)
    m = _EXTRACTIVE_RE.search(raw)
    if m:
        idx = int(m.group(1)) - 1
        evidence = m.group(2).strip().strip('"').strip("'")
        quote = m.group(3).strip().strip('"').strip("'")
        if quote and quote.lower() not in {"none", "n/a"}:
            ordered = []
            if 0 <= idx < len(passages):
                ordered.append(passages[idx])
            ordered += [p for i, p in enumerate(passages) if i != idx]
            answer = _find_verbatim(quote, ordered) or quote
            located_evidence = _find_verbatim(evidence, ordered) or evidence
            return answer, located_evidence

    # Stage 2: EVIDENCE + ANSWER without PASSAGE number
    m2 = _EVIDENCE_ANSWER_RE.search(raw)
    if m2:
        evidence = m2.group(1).strip().strip('"').strip("'")
        quote = m2.group(2).strip().strip('"').strip("'")
        if quote and quote.lower() not in {"none", "n/a"}:
            answer = _find_verbatim(quote, passages) or quote
            located_evidence = _find_verbatim(evidence, passages) or evidence
            return answer, located_evidence

    # Stage 3: just ANSWER line — no evidence from LLM
    m3 = _ANSWER_ONLY_RE.search(raw)
    if m3:
        quote = m3.group(1).strip().strip('"').strip("'")
        if quote and quote.lower() not in {"none", "n/a"}:
            answer = _find_verbatim(quote, passages) or quote
            return answer, None

    return None


def llm_generate(question: str, context_chunks: list[str]) -> tuple[str, str | None]:
    client = get_llm_client()

    # Extractive: ask LLM to select passage, evidence sentence, and answer verbatim
    messages = build_extractive_prompt(question, context_chunks)
    resp = client.chat.completions.create(
        model=LLM_MODEL_ID, messages=messages, max_tokens=256)
    raw = resp.choices[0].message.content.strip()
    result = _parse_extractive(raw, context_chunks)
    if result:
        answer, evidence = result
        if evidence is not None:
            return answer, evidence  # Stage 1/2: full result, trust answer
        # Stage 3: only ANSWER matched, evidence=None — answer quality unreliable,
        # fall through to generative for a better answer

    # Fallback: generative (better answer quality via few-shot prompt)
    messages = build_prompt(question, context_chunks)
    resp = client.chat.completions.create(
        model=LLM_MODEL_ID, messages=messages, max_tokens=128)
    return _strip_preface(resp.choices[0].message.content.strip()), None


# ── Core Pipeline ─────────────────────────────────────────────────────────────

def process_single_paper(
    item: dict,
    cfg: dict | None = None,
    verbose: bool = True,
    vectorstore=None,
    parent_chunks=None,
    # unused, kept for cache-load compatibility
    child_chunks: list[str] | None = None,
) -> dict:
    if cfg is None:
        cfg = default_cfg()

    embedding_model = get_embedding_model()
    reranker = get_reranker()

    question = item["question"]
    title = item.get("title", "")
    title_prefix = f"Paper: {title}. Passage: "

    if vectorstore is None or parent_chunks is None:
        raise ValueError(
            "Vectorstore and parent_chunks must be provided (cache loading only).")

    n_chunks = vectorstore.index.ntotal
    if n_chunks == 0:
        raise ValueError(
            "Vectorstore and parent_chunks must be provided (cache loading only).")

    question = BGE_QUERY_INSTRUCTION + question

    vectorstore_k = min(cfg["TOP_K"], n_chunks)
    retrieved_docs = vectorstore.similarity_search(question, k=vectorstore_k)
    evidence_chunks = [doc.page_content.removeprefix(
        title_prefix) for doc in retrieved_docs]

    # Reranking
    pairs = [(question, chunk) for chunk in evidence_chunks]
    rerank_scores = reranker.predict(pairs)
    ranked_indices = np.argsort(rerank_scores)[::-1]
    ranked_chunks = [evidence_chunks[i] for i in ranked_indices]

    evidence_filtered = [
        c for c, i in zip(ranked_chunks, ranked_indices)
        if rerank_scores[i] > cfg["RERANK_THRESHOLD"]
    ][:cfg["EVIDENCE_TOP_K"]]
    if not evidence_filtered:
        evidence_filtered = [ranked_chunks[0]]

    del vectorstore, retrieved_docs, rerank_scores, pairs
    gc.collect()
    torch.cuda.empty_cache()

    # Select LLM input chunks (parents of top-ranked children)
    seen, llm_chunks = set(), []
    for child in ranked_chunks[:cfg["LLM_TOP_K"]]:
        parent = _find_parent_chunk(child, parent_chunks)
        if parent not in seen:
            seen.add(parent)
            llm_chunks.append(parent)

    answer, evidence_sentence = llm_generate(question, llm_chunks)

    # Use LLM-identified evidence sentence; fall back to best sentence from parent chunk
    best_evidence = evidence_sentence if evidence_sentence else _best_sentence_from_parent(
        ranked_chunks[0], parent_chunks)

    return {
        "title":           title,
        "answer":          answer,
        "evidence_chunks": [best_evidence],
        "empty":           False,
    }


# ── Run Once ──────────────────────────────────────────────────────────────────

def run_once():
    data_path = PRIVATE_DATA_PATH if USE_PRIVATE else PUBLIC_DATA_PATH
    print(f"\n{'='*60}\nUsing dataset: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total = len(dataset)
    cfg = default_cfg()
    tag = dataset_tag(data_path)
    base_dir = os.path.join(EMBED_CACHE_DIR, tag, chunk_cfg_key(cfg))

    print(f"Prebuilding embedding cache...")
    prebuild_embedding_cache(dataset, cfg, tag)

    results = []
    for idx, item in enumerate(dataset):
        title = item.get("title", f"Paper {idx+1}")
        print(f"[{idx+1}/{total}] {title[:70]}")
        try:
            cached = _load_cache(base_dir, idx)
            if cached:
                vs, pc, cc = cached
                r = process_single_paper(
                    item, cfg=cfg, vectorstore=vs, parent_chunks=pc, child_chunks=cc)
                del vs
                gc.collect()
                torch.cuda.empty_cache()
            else:
                r = process_single_paper(item, cfg=cfg)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  ⚠ CUDA OOM: {e}; skipping.")
            gc.collect()
            torch.cuda.empty_cache()
            r = {"title": title, "answer": "OOM",
                 "evidence_chunks": [], "error": "oom"}
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            r = {"title": title, "answer": "Error during processing.",
                 "evidence_chunks": [], "error": "generic"}

        if r.get("error") == "oom":
            result = {"title": title, "answer": "OOM", "evidence": ["OOM."]}
        elif r.get("error"):
            result = {
                "title": title, "answer": "Error during processing.", "evidence": ["Error."]}
        elif r.get("empty"):
            result = {"title": title, "answer": "No content found.",
                      "evidence": ["No content."]}
        else:
            result = {"title": title,
                      "answer": r["answer"], "evidence": r["evidence_chunks"]}

        result["evidence"] = result["evidence"][:40] or [
            "No evidence retrieved."]
        results.append(result)
        print(f"[{idx+1}/{total}] answer   : {result['answer'][:100]}")
        print(f"[{idx+1}/{total}] evidence : {len(result['evidence'])} chunks")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved {total} results → {OUTPUT_PATH}")

    if data_path == PUBLIC_DATA_PATH:
        print("\n" + "="*60 + "\nRunning score_public.py ...\n" + "="*60)
        subprocess.run(["uv", "run", "python", "score_public.py",
                       OUTPUT_PATH, "--dataset", data_path])


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_once()
