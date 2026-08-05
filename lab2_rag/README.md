# HW2: Document QA based on RAG

生成式人工智慧 2026 Spring · 單檔 submission：[rag.py](rag.py)

## 專案概要

基於 RAG 的論文 QA 系統。流程：

1. **清理**：移除 `BIBREF` / `INLINEFORM` / `TABREF` 等標記與多餘空白
2. **切塊**：RecursiveCharacterTextSplitter 產生 child（320/64）與 parent（1024/256）兩層 chunks
3. **檢索**：`BAAI/bge-large-en-v1.5` embedding + FAISS 相似度 Top-15（含 BGE query instruction）
4. **Rerank**：`BAAI/bge-reranker-v2-m3` 對 (question, child) pair 排序
5. **Context 組合**：取 rerank 前 `LLM_TOP_K=4` 個 child，回查其 parent chunk 作為 LLM 輸入
6. **生成**：`meta-llama/Llama-3.2-3B-Instruct`（vLLM）先做 extractive（PASSAGE / EVIDENCE / ANSWER 三段解析），若只解析到 ANSWER 則 fallback 到 few-shot 生成
7. **輸出**：`result.json`，每篇一個 `{title, answer, evidence}`

關鍵超參數在 [rag.py](rag.py) 的 Configuration 區塊。

## 檔案結構

單檔 [rag.py](rag.py)，由上而下：

| 區塊 | 內容 |
|------|------|
| Configuration | 模型名、路徑、chunk / top-k / threshold 等常數；`default_cfg()` |
| Model Initialization | vLLM OpenAI client、HuggingFace embedding、CrossEncoder reranker |
| Text Cleaning | `clean_text` 正規化 |
| Splitters | `make_splitters` 產生 child / parent splitter |
| Prompt Helpers | Generative few-shot prompt + `_strip_preface` |
| Extractive Prompt | PASSAGE / EVIDENCE / ANSWER 格式 few-shot prompt |
| Cache Helpers | `prebuild_embedding_cache` 以 `embed_cache/{tag}/{cfg_key}/paper_{i}/` 儲存 FAISS + parents/children JSON |
| Retrieval Helpers | `_find_parent_chunk`、`_best_sentence_from_parent` |
| LLM Inference | `_parse_extractive` 三階段解析、`llm_generate` 先 extractive 後 generative fallback |
| Core Pipeline | `process_single_paper` |
| Entry Point | `run_once` + `__main__` |

## 執行方式

### 1. 啟動 vLLM server

```bash
uv run vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --port 8091 \
    --api-key abc \
    --dtype float16 \
    --gpu-memory-utilization 0.4 \
    --max-model-len 25000
```

### 2. 準備資料

將 `private_dataset.json`（或 `public_dataset.json`）放在 [rag.py](rag.py) 同目錄下。

### 3. 執行

```bash
python rag.py
```

- 預設讀 `private_dataset.json`（`USE_PRIVATE=True`）；改成 public 只需把 `USE_PRIVATE` 設為 `False`
- 預設不會使用 FP16 降 model（`USE_FP16_MODELS=FALSE`）；改成需要只需把 `USE_FP16_MODELS` 設為 `True`
- 首次執行會建立 `embed_cache/` 向量快取；之後的執行會直接 reload，跳過 embedding
- 結果寫入 `result.json`，格式為 `[{title, answer, evidence}, ...]`
- 若跑的是 public dataset，會自動呼叫 `score_public.py` 計分
