# HW1 - Medical QA Fine-tuning with LLaMA 3.2-1B

## Environment

- Python 3.10+
- CUDA 12.x
- GPU: 至少 8GB VRAM（訓練用 4-bit quantization）

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

將以下兩個檔案放到 `dataset/` 資料夾：

```
dataset/
├── dataset.csv     # 訓練用資料集
└── benchmark.csv   # 最終預測用
```

## HuggingFace 存取權限

本專案使用 `meta-llama/Llama-3.2-1B-Instruct`，需要先到 HuggingFace 申請存取權限。

在 `main.py` 第 24 行替換為自己的 token：

```python
login(token="YOUR_HF_TOKEN")
```

## 執行訓練與推論
執行前需要檢查30-37行、603-610行！

```bash
python main.py
```

執行後會自動進行：

1. 資料集切分（80% train / 10% val / 10% test，`random_state=42`）
2. LoRA fine-tuning（3 epochs）
3. 在 10% test split 上推論
4. 產生 `submission_<timestamp>.csv`（benchmark 預測結果）

## 超參數

| 參數 | 值 |
|------|-----|
| Base model | `meta-llama/Llama-3.2-1B-Instruct` |
| LoRA rank / alpha | 16 / 32 |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Effective batch size | 32 (1 × grad_accum 32) |
| Epochs | 3 |
| Quantization | 4-bit NF4 (QLoRA) |
| Few-shot examples (question_id) | 4665, 7808 |
| Benchmark inference votes | 11-vote majority voting |

## 結果

訓練完成後會輸出：

- `training_results_<timestamp>.png`：Loss 與 Validation Accuracy 曲線
- `test_split_results_<timestamp>.csv`：test split 推論結果
- `submission_<timestamp>.csv`：benchmark 提交檔案

## GPU 監控

```bash
watch -n 1 nvidia-smi
```
